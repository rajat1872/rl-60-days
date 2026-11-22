from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # save to file, no window needed
import matplotlib.pyplot as plt

NUMBER_OF_ARMS = 10
NUMBER_OF_STEPS = 1000
NUMBER_OF_RUNS = 500          # average over many random bandit problems to smooth the curves
EPSILON_VALUES = [0.0, 0.01, 0.1]
SEED = 0


def pick_greedy_arm(estimated_values, random_generator):
    # Break ties randomly: np.argmax alone always picks the first tied arm
    best_arms = np.flatnonzero(estimated_values == estimated_values.max())
    return random_generator.choice(best_arms)


def run_epsilon_greedy(true_arm_means, epsilon, number_of_steps, random_generator):
    number_of_arms = len(true_arm_means)
    estimated_values = np.zeros(number_of_arms)  # Q(a): our running guess of each arm's mean reward
    pull_counts = np.zeros(number_of_arms)       # N(a): how many times we pulled each arm
    best_arm_mean = true_arm_means.max()         # q*(best arm), only used to measure regret
    regret_each_step = np.zeros(number_of_steps)
    picked_best_arm = np.zeros(number_of_steps)

    for step in range(number_of_steps):
        if random_generator.random() < epsilon:
            chosen_arm = random_generator.integers(number_of_arms)                # explore
        else:
            chosen_arm = pick_greedy_arm(estimated_values, random_generator)      # exploit

        reward = random_generator.normal(true_arm_means[chosen_arm], 1.0)

        # Incremental sample average: NewEstimate = OldEstimate + (1/n) * (Target - OldEstimate)
        pull_counts[chosen_arm] += 1
        estimated_values[chosen_arm] += (reward - estimated_values[chosen_arm]) / pull_counts[chosen_arm]

        regret_each_step[step] = best_arm_mean - true_arm_means[chosen_arm]
        picked_best_arm[step] = true_arm_means[chosen_arm] == best_arm_mean

    return np.cumsum(regret_each_step), picked_best_arm


def main():
    random_generator = np.random.default_rng(SEED)
    plt.figure(figsize=(8, 5))

    for epsilon in EPSILON_VALUES:
        cumulative_regret_all_runs = np.zeros(NUMBER_OF_STEPS)
        best_arm_rate_all_runs = np.zeros(NUMBER_OF_STEPS)
        for run_index in range(NUMBER_OF_RUNS):
            true_arm_means = random_generator.normal(0.0, 1.0, NUMBER_OF_ARMS)  # a fresh 10-armed problem
            cumulative_regret, picked_best_arm = run_epsilon_greedy(
                true_arm_means, epsilon, NUMBER_OF_STEPS, random_generator)
            cumulative_regret_all_runs += cumulative_regret
            best_arm_rate_all_runs += picked_best_arm

        average_cumulative_regret = cumulative_regret_all_runs / NUMBER_OF_RUNS
        best_arm_percentage_last_100 = 100 * best_arm_rate_all_runs[-100:].mean() / NUMBER_OF_RUNS
        print(f"epsilon {epsilon:<5} | regret after {NUMBER_OF_STEPS} steps: {average_cumulative_regret[-1]:6.1f}"
              f" | picks best arm (last 100 steps): {best_arm_percentage_last_100:4.1f}%")
        plt.plot(average_cumulative_regret, label=f"epsilon = {epsilon}")

    plt.xlabel("step")
    plt.ylabel("cumulative regret (lower is better)")
    plt.title(f"Epsilon-greedy on a {NUMBER_OF_ARMS}-armed bandit, averaged over {NUMBER_OF_RUNS} runs")
    plt.legend()
    plt.savefig(Path(__file__).with_name("regret.png"), dpi=120, bbox_inches="tight")
    print(f"saved plot to {Path(__file__).with_name('regret.png')}")


if __name__ == "__main__":
    # Self-check: the incremental update must equal the plain average
    running_estimate, count = 0.0, 0
    for sample_reward in [2.0, 4.0, 9.0]:
        count += 1
        running_estimate += (sample_reward - running_estimate) / count
    assert abs(running_estimate - 5.0) < 1e-12
    main()
