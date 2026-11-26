import random
import gymnasium as gym
from gymnasium.utils.env_checker import check_env

import config_fix_env  # registers ConfigFix-v0
from config_fix_env import FIELD_NAMES, GOOD_DEFAULTS


def oracle_agent(observation):
    # Reads the last tool output and acts like a careful engineer: check, fix each failing field, submit
    last_tool_output = observation.split("---\n")[-1]
    if last_tool_output.startswith("failing: "):
        first_failing_field = last_tool_output[len("failing: "):].split(", ")[0]
        return f"set {first_failing_field} {GOOD_DEFAULTS[first_failing_field]}"
    if last_tool_output == "all checks pass":
        return "submit"
    return "check"


def make_random_agent(random_generator):
    plausible_values = ["8080", "80", "INFO", "VERBOSE", "30", "-5", "3", "0", "true", "yes"]

    def random_agent(observation):
        return random_generator.choice(
            ["check", "submit"] + [f"set {random_generator.choice(FIELD_NAMES)} {random_generator.choice(plausible_values)}"] * 3)
    return random_agent


def run_episode(environment, agent, seed, print_trace=False):
    observation, info = environment.reset(seed=seed)
    total_reward, step_count = 0.0, 0
    if print_trace:
        print(f"[reset seed={seed}] hidden info: {info}\n{observation}\n")
    while True:
        action = agent(observation)
        observation, reward, terminated, truncated, info = environment.step(action)
        total_reward += reward
        step_count += 1
        if print_trace:
            last_tool_output = observation.split("---\n")[-1]
            print(f"step {step_count}: action={action!r:28} -> {last_tool_output!r:40} reward={reward} "
                  f"terminated={terminated} truncated={truncated}")
        if terminated or truncated:
            return total_reward, step_count


def test_determinism():
    # Same seed + same actions must give byte-identical observations and rewards
    actions = ["check", "set port 8080", "check", "submit"]
    transcripts = []
    for _ in range(2):
        environment = gym.make("ConfigFix-v0")
        observation, info = environment.reset(seed=123)
        transcript = [observation]
        for action in actions:
            observation, reward, terminated, truncated, info = environment.step(action)
            transcript.append((observation, reward, terminated, truncated))
        transcripts.append(transcript)
    assert transcripts[0] == transcripts[1], "same seed produced different episodes"
    different_seed_environment = gym.make("ConfigFix-v0")
    assert different_seed_environment.reset(seed=124)[0] != transcripts[0][0], "seed has no effect"
    print("determinism test: PASS")


def test_reward_cannot_be_gamed_by_partial_fix():
    environment = gym.make("ConfigFix-v0")
    environment.reset(seed=7)
    environment.unwrapped.config = dict(GOOD_DEFAULTS, replicas="0")  # one field still broken
    _, reward, terminated, _, _ = environment.step("submit")
    assert reward == 0.0 and terminated
    print("partial-fix test: PASS")


if __name__ == "__main__":
    check_env(gym.make("ConfigFix-v0").unwrapped)   # Gymnasium's own API-contract checker
    print("gymnasium check_env: PASS")
    test_determinism()
    test_reward_cannot_be_gamed_by_partial_fix()

    environment = gym.make("ConfigFix-v0")
    print("\n=== oracle agent, one traced episode ===")
    run_episode(environment, oracle_agent, seed=42, print_trace=True)

    for agent_name, agent in [("oracle", oracle_agent), ("random", make_random_agent(random.Random(0)))]:
        results = [run_episode(environment, agent, seed) for seed in range(200)]
        success_rate = sum(reward for reward, _ in results) / len(results)
        average_steps = sum(steps for _, steps in results) / len(results)
        print(f"{agent_name:6s} agent over 200 seeds: success {success_rate:.1%}, average steps {average_steps:.1f}")
