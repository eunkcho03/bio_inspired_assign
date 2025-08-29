# ===================== DQN PART (drop-in) =====================
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import collections as cl
from tqdm import tqdm
up, right, down, left = 0, 1, 2, 3
actions = [up, right, down, left]
directions = {up: (0, -1), right: (1, 0), down: (0, 1), left: (-1, 0)}


class CNN(nn.Module):
    def __init__(self, in_channels, h, w, n_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.lin = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  
            nn.Flatten(),             
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, n_actions),
        )

    def forwardpass(self, x):
        x = self.conv(x)
        x = self.lin(x)
        return x


class Replay_Buffer:
    def __init__(self, capacity, seed=None):
        self.buf = cl.deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def push(self, obs, action, reward, next_obs, done):
        self.buf.append({
            "obs":      obs.astype(np.float32),
            "action":   int(action),
            "reward":   float(reward),
            "next_obs": next_obs.astype(np.float32),
            "done":     float(done),
        })

    def sample(self, batch_size):
        batch = self.rng.sample(self.buf, batch_size)
        obs      = np.stack([b["obs"]      for b in batch], axis=0)
        next_obs = np.stack([b["next_obs"] for b in batch], axis=0)
        actions_ = np.asarray([b["action"] for b in batch], dtype=np.int64)
        rewards  = np.asarray([b["reward"] for b in batch], dtype=np.float32)
        dones    = np.asarray([b["done"]   for b in batch], dtype=np.float32)
        return obs, actions_, rewards, next_obs, dones

    def __len__(self):
        return len(self.buf)


class Training:
    def __init__(self, env, total_env_steps, buffer_cap, batch_size, gamma, lr, min_buffer, target_update_every, eval_every, seed, save_path, eps_start, eps_end, eps_fraction):
        self.env = env
        self.total_env_steps = total_env_steps
        self.buffer_cap = buffer_cap
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.min_buffer = min_buffer
        self.target_update_every = target_update_every
        self.eval_every = eval_every
        self.save_path = save_path
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.decay_steps = int(max(1, eps_fraction * total_env_steps))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if seed is not None:
            torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

        C, H, W = env.observation_tensor().shape
        nA = len(actions)

        self.online = CNN(C, H, W, nA).to(self.device)
        self.target = CNN(C, H, W, nA).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.opt = optim.Adam(self.online.parameters(), lr=self.lr)
        self.rb = Replay_Buffer(self.buffer_cap, seed)

    def epsilon_value_linear(self, t):
        frac = min(1.0, t / max(1, self.decay_steps))
        return self.eps_start + (self.eps_end - self.eps_start) * frac

    def select_action(self, obs_np, step):
        eps = self.epsilon_value_linear(step)
        if random.random() < eps:
            return random.choice(actions), eps
        with torch.no_grad():
            obs_t = torch.from_numpy(obs_np).unsqueeze(0).to(self.device) 
            q = self.online.forwardpass(obs_t)
            a = int(q.argmax(dim=1).item())
            return a, eps

    @torch.no_grad()
    def compute_td_targets(self, next_obs_b, rewards_b, dones_b):
        next_q_online = self.online.forwardpass(next_obs_b)                           
        next_actions  = next_q_online.argmax(dim=1, keepdim=True)                        
        next_q_target = self.target.forwardpass(next_obs_b).gather(1, next_actions).squeeze(1)  
        return rewards_b + self.gamma * (1.0 - dones_b) * next_q_target

    @torch.no_grad()
    def evaluate_policy(self, n_episodes=10):
        self.online.eval()
        total_return, successes = 0.0, 0
        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            ep_ret = 0.0
            for _ in range(int(self.env.cfg["max_steps"])):
                obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.device)
                q = self.online.forwardpass(obs_t)
                a = int(q.argmax(dim=1).item())
                obs, r, done, info = self.env.step(a)
                ep_ret += r
                if done:
                    if info.get("success", False): successes += 1
                    break
            total_return += ep_ret
        return successes / n_episodes, total_return / n_episodes

    def train_dqn(self):
        obs, _ = self.env.reset()
        ep_return, ep_len = 0.0, 0
        best_success = 0.0

        losses_win = cl.deque(maxlen=200)
        returns_win = cl.deque(maxlen=100)

        pbar = tqdm(range(1, self.total_env_steps + 1), desc="Train(DQN)", dynamic_ncols=True)
        for step in pbar:
            a, eps = self.select_action(obs, step)
            nxt, r, done, info = self.env.step(a)

            self.rb.push(obs, a, r, nxt, done)
            ep_return += r
            ep_len += 1
            obs = nxt

            loss_val = None
            if len(self.rb) >= self.min_buffer:
                obs_np, acts_np, rews_np, next_obs_np, dones_np = self.rb.sample(self.batch_size)

                obs_b      = torch.from_numpy(obs_np).to(self.device)         # (B,C,H,W)
                next_obs_b = torch.from_numpy(next_obs_np).to(self.device)    # (B,C,H,W)
                acts_b     = torch.from_numpy(acts_np).to(self.device)        # (B,)
                rews_b     = torch.from_numpy(rews_np).to(self.device)        # (B,)
                dones_b    = torch.from_numpy(dones_np).to(self.device)       # (B,)

                q_pred = self.online.forwardpass(obs_b).gather(1, acts_b.view(-1,1)).squeeze(1)  # (B,)
                targets = self.compute_td_targets(next_obs_b, rews_b, dones_b)

                loss = nn.functional.smooth_l1_loss(q_pred, targets)
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.online.parameters(), 1.0)
                self.opt.step()
                loss_val = float(loss.item())
                losses_win.append(loss_val)

            if done:
                returns_win.append(ep_return)
                obs, _ = self.env.reset()
                ep_return, ep_len = 0.0, 0

            if step % self.target_update_every == 0:
                self.target.load_state_dict(self.online.state_dict())

            if step % self.eval_every == 0:
                succ, avg_ret = self.evaluate_policy(n_episodes=20)
                print(f"[step {step}] success={succ:.2%} avg_return={avg_ret:.2f} "
                      f"| buffer={len(self.rb)} | eps={eps:.3f}")
                if succ > best_success:
                    best_success = succ
                    torch.save(self.online.state_dict(), self.save_path)
                    print(f"  ↳ New best success, model saved to {self.save_path}")

            avg_loss = (sum(losses_win) / len(losses_win)) if len(losses_win) else None
            avg_ret_recent = (sum(returns_win) / len(returns_win)) if len(returns_win) else 0.0
            pbar.set_postfix({
                "loss": f"{avg_loss:.3f}" if avg_loss is not None else "warmup",
                "eps": f"{eps:.3f}",
                "buf": len(self.rb),
                "ret": f"{avg_ret_recent:.1f}",
            })

        torch.save(self.online.state_dict(), self.save_path)
        return self.online, self.target, self.save_path

    def make_policy_fn(self, model=None):
        model = model if model is not None else self.online
        dev = self.device
        model.eval()
        def policy_fn(env, obs):
            with torch.no_grad():
                obs_t = torch.from_numpy(obs).unsqueeze(0).to(dev)
                q = model.forwardpass(obs_t)
                return int(q.argmax(dim=1).item())
        return policy_fn
