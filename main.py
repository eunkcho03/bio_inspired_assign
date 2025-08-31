import yaml
from Environment_Setup import Environment, visualize_episode_pg, visualize_snapshot_pg
from Dqn import Training, record_greedy_episode
from Evaluation import plot_training_curves

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
        eps_start=eps_cfg["start"],
        eps_end=eps_cfg["end"],
        eps_fraction=eps_cfg["fraction"],
        plot_every= train_cfg["plot_every"],
    )    
    
    trainer.train_dqn()
    trainer.save_snapshots("snapshots.pth")
    trainer.make_trajec(direct="plots", visualize_pg=True)
    trainer.save_metrics_excel("plots/metrics.xlsx")
    plot_training_curves(trainer.plotting)
    greedy_policy = trainer.make_policy_fn() 
    final_path = record_greedy_episode(env, greedy_policy)
    visualize_snapshot_pg(env, final_path,title="Greedy Policy (Final)",save_path="plots/greedy_final_pg.png")
    success_rate, avg_return, avg_steps, best_steps, avg_path_length = trainer.evaluate_policy(n_episodes=50)
    print(f"FINAL RESULT: success_rate={success_rate:.2%}, avg_return={avg_return:.2f}, avg_steps= {avg_steps:.2f}, best_steps={best_steps}, average_path_length {avg_path_length} ")

if __name__ == "__main__":
    main()