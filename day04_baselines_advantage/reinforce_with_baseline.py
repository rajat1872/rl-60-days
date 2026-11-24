import sys
from pathlib import Path

import gymnasium as gym
import torch
import torch.nn as nn
from torch.distributions import Categorical

NUMBER_OF_EPISODES = 1000
DISCOUNT_FACTOR = 0.99
POLICY_LEARNING_RATE = 1e-3
VALUE_LEARNING_RATE = 1e-2
HIDDEN_SIZE = 128


# Same policy network and return computation as Day 3, copied so this script runs on its own
class PolicyNetwork(nn.Module):
    # Maps a CartPole state (4 numbers) to a score (logit) for each action (left, right)
    def __init__(self, state_size, number_of_actions, hidden_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, number_of_actions),
        )

    def forward(self, state):
        return Categorical(logits=self.layers(state))  # pi(a|s) as a probability distribution


def compute_returns_for_every_step(reward_list, discount_factor):
    # G_t for every t, computed backwards: G_t = r_{t+1} + gamma * G_{t+1}
    returns = []
    running_return = 0.0
    for reward in reversed(reward_list):
        running_return = reward + discount_factor * running_return
        returns.append(running_return)
    returns.reverse()
    return torch.tensor(returns)


class ValueNetwork(nn.Module):
    # The baseline / critic: maps a state to one number, its predicted return V(s)
    def __init__(self, state_size, hidden_size):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(state_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1))

    def forward(self, states):
        return self.layers(states).squeeze(-1)


def run_one_episode_recording_states(environment, policy_network, seed):
    state, info = environment.reset(seed=seed)
    visited_states, log_probabilities_of_taken_actions, rewards = [], [], []
    episode_is_over = False
    while not episode_is_over:
        state_tensor = torch.as_tensor(state, dtype=torch.float32)
        action_distribution = policy_network(state_tensor)
        action = action_distribution.sample()
        visited_states.append(state_tensor)
        log_probabilities_of_taken_actions.append(action_distribution.log_prob(action))
        state, reward, terminated, truncated, info = environment.step(action.item())
        rewards.append(reward)
        episode_is_over = terminated or truncated
    return torch.stack(visited_states), torch.stack(log_probabilities_of_taken_actions), rewards


def train(use_baseline, seed):
    torch.manual_seed(seed)
    environment = gym.make("CartPole-v1")
    policy_network = PolicyNetwork(state_size=4, number_of_actions=2, hidden_size=HIDDEN_SIZE)
    value_network = ValueNetwork(state_size=4, hidden_size=HIDDEN_SIZE)
    policy_optimizer = torch.optim.Adam(policy_network.parameters(), lr=POLICY_LEARNING_RATE)
    value_optimizer = torch.optim.Adam(value_network.parameters(), lr=VALUE_LEARNING_RATE)
    episode_returns = []

    for episode_index in range(NUMBER_OF_EPISODES):
        states, log_probabilities, rewards = run_one_episode_recording_states(
            environment, policy_network, seed=seed * 10_000 + episode_index)
        returns = compute_returns_for_every_step(rewards, DISCOUNT_FACTOR)

        if use_baseline:
            predicted_values = value_network(states)
            # Advantage = how much better than expected. detach(): the policy must not push the critic around
            advantages = returns - predicted_values.detach()
            value_loss = ((predicted_values - returns) ** 2).mean()   # critic learns to predict G_t
            value_optimizer.zero_grad()
            value_loss.backward()
            value_optimizer.step()
        else:
            advantages = returns                                        # plain REINFORCE (Day 3)

        policy_loss = -(log_probabilities * advantages).sum()
        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()
        episode_returns.append(sum(rewards))

    environment.close()
    return episode_returns


if __name__ == "__main__":
    use_baseline_flag = sys.argv[1] == "baseline"
    seed_value = int(sys.argv[2])
    all_returns = train(use_baseline_flag, seed_value)
    torch.save(all_returns, Path(__file__).with_name(f"returns_{sys.argv[1]}_seed{seed_value}.pt"))
    print(sys.argv[1], seed_value, "last-100 average return:", sum(all_returns[-100:]) / 100)
