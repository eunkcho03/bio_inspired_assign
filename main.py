import yaml
from Environment_Setup import Environment, visualize_episode_pg
from Dqn import Training

env_cfg_path = "config.yaml"
train_cfg_path = "train_config.yaml"

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f) 
    

def main():
    env_cfg = load_yaml(env_cfg_path)
    train_cfg = load_yaml(train_cfg_path)
    
    env = Environment(env_cfg)
    eps_cfg = train_cfg["eps"]
    trainer = Training(
        env,
        total_env_steps=train_cfg["total_env_steps"],
        buffer_cap=train_cfg["buffer_cap"],
        batch_size=train_cfg["batch_size"],
        gamma=train_cfg["gamma"],
        lr=train_cfg["lr"],
        min_buffer=train_cfg["min_buffer"],
        target_update_every=train_cfg["target_update_every"],
        eval_every=train_cfg["eval_every"],
        seed=env_cfg["seed"] ,
        save_path=train_cfg["save_path"],
        eps_start=eps_cfg["start"],
        eps_end=eps_cfg["end"],
        eps_fraction=eps_cfg["fraction"],
    )    
    
    trainer.train_dqn()
    
    print("hello")
    success_rate, avg_return = trainer.evaluate_policy(n_episodes=50)
    print(f"FINAL RESULT → success_rate={success_rate:.2%}, avg_return={avg_return:.2f}")

    
    visualize_episode_pg(env, policy_fn=trainer.make_policy_fn(), fps=8, max_steps=300, scale=32)

if __name__ == "__main__":
    main()