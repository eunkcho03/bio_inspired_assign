import matplotlib.pyplot as plt
import os

def plot_training_curves(metrics, outdir="plots"):
    os.makedirs(outdir, exist_ok=True)

    steps = metrics["step"]
    avg_returns = metrics["avg_return"]
    success = metrics["success_rate"]

    plt.figure()
    plt.plot(steps, avg_returns, label="Avg return")
    plt.xlabel("Training steps")
    plt.ylabel("Average return")
    plt.grid(True)
    plt.tight_layout()
    save_path = os.path.join(outdir, "training_avg_return.png")
    plt.savefig(save_path, dpi=160)
    plt.close()
    print(f"Saved: {save_path}")

    plt.figure()
    plt.plot(steps, success, label="Success rate")
    plt.xlabel("Training steps")
    plt.ylabel("Success rate")
    plt.grid(True)
    plt.tight_layout()
    save_path = os.path.join(outdir, "training_success_rate.png")
    plt.savefig(save_path, dpi=160)
    plt.close()
    print(f"Saved: {save_path}")