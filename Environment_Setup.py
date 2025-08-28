from __future__ import annotations
import numpy as np
import random
import yaml

##########################Setup Environment <3##########################
up, right, down, left = 0, 1, 2, 3
actions = [up, right, down, left]
directions = {up: (0, -1), right: (1, 0), down: (0, 1), left: (-1, 0)}

T_emp = 0
T_wall = 1
T_trap = 2
t_slide = 3
T_decoy = 4
T_portal = 5
T_depot = 6
T_pickup = 7

class Environment:
    def __init__(self, cfg):
        self.cfg = cfg
        self.rng = random.Random(cfg.get("seed"))
        self.np_rng = np.random.default_rng(cfg.get("seed"))
        self.grid, self.picks_all, self.depot_pos, self.portal_map = self.ascii(cfg["ascii_map"])
        self.H = len(self.grid)
        self.W = len(self.grid[0])
        self.agent_start = self.depot_pos
        self.reset()
        
    def reset(self):
        self.agent_pos = self.agent_start
        self.picks_remaining = set(self.picks_all)
        self.step_count = 0
        return self.observation_tensor(), {}
    
    def ascii(self, ascii_map):  # the input is string
        lines = []
        for l in ascii_map.strip().split("\n"):
            lines.append(l)
        H = len(lines)
        W = len(lines[0])
        grid = []
        for _ in range(H):
            grid.append([T_emp] * W)
        picks_all = set()
        portals = []
        depot_pos = (0, 0)
        portal_map = {}              # <-- MOVED HERE (before the loops)

        for r in range(H):
            for c in range(W):
                ch = lines[r][c]
                if ch == 'X':
                    grid[r][c] = T_wall
                elif ch == 'T':
                    grid[r][c] = T_trap
                elif ch == 'W':
                    grid[r][c] = t_slide
                elif ch == 'D':
                    grid[r][c] = T_decoy
                elif ch == 'H':
                    grid[r][c] = T_depot
                    depot_pos = (r, c)
                elif ch == 'P':
                    grid[r][c] = T_portal
                    portals.append((r, c))
                elif ch.isdigit():
                    grid[r][c] = T_pickup
                    picks_all.add((r, c))

        for i in range(0, len(portals), 2):  
            a = portals[i]
            b = portals[i + 1]
            portal_map[a] = b
            portal_map[b] = a

        return grid, picks_all, depot_pos, portal_map

                
    def observation_tensor(self):
        layers = []
        
        # Agent layer
        agent_layer = np.zeros((self.H, self.W), dtype=np.float32)
        agent_layer[self.agent_pos] = 1.0
        layers.append(agent_layer)
        
        # Depot layer
        depot_layer = np.zeros((self.H, self.W), dtype=np.float32)
        depot_layer[self.depot_pos] = 1.0
        layers.append(depot_layer)
        
        # Pickup Layer
        pickup_layer = np.zeros((self.H, self.W), dtype=np.float32)
        for p in self.picks_remaining:
            pickup_layer[p] = 1.0
        layers.append(pickup_layer)
        
        Tiles = [T_trap, t_slide, T_decoy, T_portal]
        
        for tile in Tiles:
            tile_layer = (np.array(self.grid) == tile).astype(np.float32)
            layers.append(tile_layer)
        
        observation = np.stack(layers, axis=0)
        return observation
    
    def step(self,action):
        r, c = self.agent_pos
        dx, dy = directions[action]
        nr = r + dy
        nc = c + dx
        if not (0<=nr<self.H and 0 <=nc <self.W) or self.grid[nr][nc] == T_wall:
            nr, nc = r, c
        pos = (nr, nc)
        
        # In case of slide
        if self.grid[pos[0]][pos[1]]== t_slide and self.rng.random() < float(self.cfg["p_stochastic"]):
            if action in (up, down):
                side_choices = [left, right]
            else:
                side_choices = [up, down]
            slip_action = self.rng.choice(side_choices)
            sdx, sdy = directions[slip_action]
            sr, sc = pos[0] + sdy, pos[1] + sdx
            
            if 0 <= sr <self.H and 0<= sc < self.W and self.grid[sr][sc] != T_wall:
                pos = (sr, sc)
        
        # In case of portal
        teleported = False
        if self.grid[pos[0]][pos[1]] == T_portal:
            dest = self.portal_map.get(pos)
            if dest is not None:
                pos = dest
                teleported = True
    
        self.agent_pos = pos
        
        reward = float(self.cfg["c_step"])
        if teleported:
            reward += float(self.cfg["c_portal"])
        
        if self.agent_pos in self.picks_remaining:
            self.picks_remaining.remove(self.agent_pos)
            reward += float(self.cfg["r_item"])
        
        if self.grid[self.agent_pos[0]][self.agent_pos[1]] == T_decoy:
            d = self.cfg["decoy"]
            p = float(d["p_stochastic"])
            if self.rng.random() < p:
                reward += float(d["r_good"]) 
            else:
                reward += float(d["r_bad"])
                
        if self.grid[self.agent_pos[0]][self.agent_pos[1]] == T_trap:
            reward += float(self.cfg["r_trap"])
            
        done = False
        
        success = (len(self.picks_remaining)==0 and self.agent_pos == self.depot_pos)
        if success:
            reward += float(self.cfg["complete_bonus"])
            done = True
        self.step_count += 1
        if self.step_count >= int(self.cfg["max_steps"]):
            done = True
        
        return self.observation_tensor(), reward, done, {"success": success, "teleported": teleported}
    
    def render_ascii(self):
        # Characters per tile
        chars = {
            T_emp:'.', T_wall:'X', T_trap:'T', t_slide:'W',
            T_decoy:'D', T_portal:'P', T_depot:'H', T_pickup:'.'
        }
        lines = []
        for r in range(self.H):
            row = []
            for c in range(self.W):
                ch = chars[self.grid[r][c]]
                if (r, c) in self.picks_remaining: ch = '*'
                if (r, c) == self.depot_pos:       ch = 'H'
                row.append(ch)
            lines.append(''.join(row))
        # overlay agent
        ar, ac = self.agent_pos
        ln = list(lines[ar]); ln[ac] = '@'; lines[ar] = ''.join(ln)
        print('\n'.join(lines))



import pygame as pg

def draw_env_pg(screen, env, scale=32):
    colors = {
        T_emp:(230,230,230), T_wall:(30,30,30),   T_trap:(200,60,60),
        t_slide:(80,160,220), T_decoy:(200,120,60), T_portal:(130,70,200),
        T_depot:(60,180,90)
    }
    H, W = env.H, env.W
    for r in range(H):
        for c in range(W):
            tile = env.grid[r][c]
            col = colors.get(tile, (200,200,200))
            pg.draw.rect(screen, col, (c*scale, r*scale, scale, scale))
            # picks overlay
            if (r,c) in env.picks_remaining:
                pg.draw.circle(screen, (255,215,0), (c*scale+scale//2, r*scale+scale//2), scale//5)
    # depot marker
    dr, dc = env.depot_pos
    pg.draw.rect(screen, (60,180,90), (dc*scale, dr*scale, scale, scale), width=3)
    # agent
    ar, ac = env.agent_pos
    pg.draw.circle(screen, (0,0,0), (ac*scale+scale//2, ar*scale+scale//2), scale//3)

def visualize_episode_pg(env, policy_fn=None, fps=8, max_steps=500, scale=32):
    """policy_fn(env, obs) -> action (0..3). If None, random actions."""
    pg.init()
    screen = pg.display.set_mode((env.W*scale, env.H*scale))
    clock = pg.time.Clock()
    obs, info = env.reset()
    running = True
    steps = 0
    while running and steps < max_steps:
        for event in pg.event.get():
            if event.type == pg.QUIT: running = False

        a = policy_fn(env, obs) if policy_fn else random.choice([0,1,2,3])
        obs, r, done, info = env.step(a)

        screen.fill((240,240,240))
        draw_env_pg(screen, env, scale)
        pg.display.flip()
        clock.tick(fps)

        if done:
            # brief flash on success
            if info.get("success"):
                screen.fill((180,255,180)); pg.display.flip(); pg.time.delay(300)
            obs, info = env.reset()
        steps += 1
    pg.quit()


with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)
    
# Build env
env = Environment(cfg)

# 1) Reset + ASCII render
obs, info = env.reset()
env.render_ascii()

# 2) Take a few random steps and render
for _ in range(5):
    a = random.choice([0,1,2,3])
    obs, r, done, info = env.step(a)
    env.render_ascii()
    print("r:", r, "done:", done, "info:", info)
    if done:
        obs, info = env.reset()

# 3) Pygame viewer (random policy)
visualize_episode_pg(env, fps=8, max_steps=300, scale=32)
