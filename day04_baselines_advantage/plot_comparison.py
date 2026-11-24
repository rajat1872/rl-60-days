from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEEDS = [0, 1, 2]
WINDOW = 50
HERE = Path(__file__).parent

plt.figure(figsize=(8, 5))
for variant, colour in [("plain", "tab:blue"), ("baseline", "tab:green")]:
    smoothed_runs = []
    for seed in SEEDS:
        episode_returns = np.array(torch.load(HERE / f"returns_{variant}_seed{seed}.pt"))
        smoothed = np.convolve(episode_returns, np.ones(WINDOW) / WINDOW, mode="valid")  # moving average
        smoothed_runs.append(smoothed)
        plt.plot(smoothed, color=colour, alpha=0.25, linewidth=1)
        print(f"{variant:8s} seed {seed}: last-100 avg {episode_returns[-100:].mean():6.1f} | "
              f"first episode reaching 500: {int(np.argmax(episode_returns >= 500)) if (episode_returns >= 500).any() else 'never'}")
    label = "REINFORCE (Day 3)" if variant == "plain" else "REINFORCE + learned baseline (Day 4)"
    plt.plot(np.mean(smoothed_runs, axis=0), color=colour, linewidth=2.5, label=label)

plt.xlabel("episode")
plt.ylabel(f"return ({WINDOW}-episode moving average)")
plt.title("CartPole: thin lines = 3 individual seeds, thick = their mean")
plt.legend()
plt.savefig(HERE / "baseline_comparison.png", dpi=120, bbox_inches="tight")
print("saved baseline_comparison.png")
