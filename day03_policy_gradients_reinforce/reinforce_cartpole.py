import gymnasium as gym
import torch
import torch.nn as nn
from torch.distributions import Categorical

NUMBER_OF_EPISODES = 1000
DISCOUNT_FACTOR = 0.99
LEARNING_RATE = 1e-3
HIDDEN_SIZE = 128
SEED = 0


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


def run_one_episode(environment, policy_network, seed):
    state, info = environment.reset(seed=seed)
    log_probabilities_of_taken_actions = []
    rewards = []
    episode_is_over = False
    while not episode_is_over:
        action_distribution = policy_network(torch.as_tensor(state, dtype=torch.float32))
        action = action_distribution.sample()                        # explore by sampling, not argmax
        log_probabilities_of_taken_actions.append(action_distribution.log_prob(action))
        state, reward, terminated, truncated, info = environment.step(action.item())
        rewards.append(reward)
        episode_is_over = terminated or truncated
    return torch.stack(log_probabilities_of_taken_actions), rewards


def main():
    torch.manual_seed(SEED)
    environment = gym.make("CartPole-v1")
    policy_network = PolicyNetwork(state_size=4, number_of_actions=2, hidden_size=HIDDEN_SIZE)
    optimizer = torch.optim.Adam(policy_network.parameters(), lr=LEARNING_RATE)
    recent_episode_lengths = []

    for episode_index in range(NUMBER_OF_EPISODES):
        log_probabilities, rewards = run_one_episode(environment, policy_network, seed=SEED + episode_index)
        returns = compute_returns_for_every_step(rewards, DISCOUNT_FACTOR)

        # REINFORCE: gradient ascent on sum_t G_t * log pi(a_t|s_t)  ==  gradient descent on its negative
        loss = -(log_probabilities * returns).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        recent_episode_lengths.append(len(rewards))
        if (episode_index + 1) % 50 == 0:
            average_length = sum(recent_episode_lengths[-50:]) / 50
            print(f"episodes {episode_index - 48:4d}-{episode_index + 1:4d} | average return {average_length:6.1f}")

    environment.close()


if __name__ == "__main__":
    assert torch.allclose(compute_returns_for_every_step([0, 0, 10], 0.9), torch.tensor([8.1, 9.0, 10.0]))
    main()
