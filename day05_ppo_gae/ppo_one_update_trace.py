# One PPO rollout + one update on CartPole, written to mirror CleanRL ppo.py lines 192-290
# (single env instead of 4, value-loss clipping off) so every number can be printed and followed.
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

ROLLOUT_STEPS = 128          # CleanRL num_steps
DISCOUNT_FACTOR = 0.99       # gamma
GAE_LAMBDA = 0.95
UPDATE_EPOCHS = 4
NUMBER_OF_MINIBATCHES = 4
CLIP_COEFFICIENT = 0.2
VALUE_COEFFICIENT = 0.5
ENTROPY_COEFFICIENT = 0.01
LEARNING_RATE = 2.5e-4
MAX_GRADIENT_NORM = 0.5
SEED = 1


def compute_gae(rewards, values, episode_ended_after_step, value_after_last_step, gamma, gae_lambda):
    # Walk backwards: delta_t = r_t + gamma*V(s_{t+1})*(not done) - V(s_t);  A_t = delta_t + gamma*lambda*(not done)*A_{t+1}
    advantages = torch.zeros_like(rewards)
    running_advantage = 0.0
    for step in reversed(range(len(rewards))):
        next_state_value = value_after_last_step if step == len(rewards) - 1 else values[step + 1]
        not_done = 1.0 - episode_ended_after_step[step]
        delta = rewards[step] + gamma * next_state_value * not_done - values[step]
        running_advantage = delta + gamma * gae_lambda * not_done * running_advantage
        advantages[step] = running_advantage
    return advantages


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.critic = nn.Sequential(nn.Linear(4, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))
        self.actor = nn.Sequential(nn.Linear(4, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 2))

    def get_action_and_value(self, states, actions=None):
        distribution = Categorical(logits=self.actor(states))
        if actions is None:
            actions = distribution.sample()
        return actions, distribution.log_prob(actions), distribution.entropy(), self.critic(states).squeeze(-1)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    environment = gym.make("CartPole-v1")
    agent = ActorCritic()
    optimizer = torch.optim.Adam(agent.parameters(), lr=LEARNING_RATE, eps=1e-5)

    # ---- 1. ROLLOUT: run the current policy for 128 steps, store everything (no gradients) ----
    states = torch.zeros(ROLLOUT_STEPS, 4)
    actions = torch.zeros(ROLLOUT_STEPS, dtype=torch.long)
    old_log_probabilities = torch.zeros(ROLLOUT_STEPS)
    rewards = torch.zeros(ROLLOUT_STEPS)
    episode_ended_after_step = torch.zeros(ROLLOUT_STEPS)
    old_values = torch.zeros(ROLLOUT_STEPS)
    state, info = environment.reset(seed=SEED)
    for step in range(ROLLOUT_STEPS):
        states[step] = torch.as_tensor(state, dtype=torch.float32)
        with torch.no_grad():
            action, log_probability, _, value = agent.get_action_and_value(states[step])
        actions[step], old_log_probabilities[step], old_values[step] = action, log_probability, value
        state, reward, terminated, truncated, info = environment.step(action.item())
        rewards[step] = reward
        episode_ended_after_step[step] = float(terminated or truncated)
        if terminated or truncated:
            state, info = environment.reset()
    with torch.no_grad():
        value_after_last_step = agent.get_action_and_value(torch.as_tensor(state, dtype=torch.float32))[3]

    # ---- 2. ADVANTAGES with GAE, and value targets ----
    advantages = compute_gae(rewards, old_values, episode_ended_after_step, value_after_last_step,
                             DISCOUNT_FACTOR, GAE_LAMBDA)
    value_targets = advantages + old_values
    print(f"rollout: {ROLLOUT_STEPS} steps, {int(episode_ended_after_step.sum())} episodes ended inside it")
    print("first 4 steps  | value", [round(v, 3) for v in old_values[:4].tolist()],
          "| advantage", [round(a, 2) for a in advantages[:4].tolist()],
          "| target", [round(t, 2) for t in value_targets[:4].tolist()])

    # ---- 3. UPDATE: 4 epochs x 4 shuffled minibatches of 32 on the SAME rollout ----
    minibatch_size = ROLLOUT_STEPS // NUMBER_OF_MINIBATCHES
    all_indices = np.arange(ROLLOUT_STEPS)
    for epoch in range(UPDATE_EPOCHS):
        np.random.shuffle(all_indices)
        for minibatch_start in range(0, ROLLOUT_STEPS, minibatch_size):
            minibatch = all_indices[minibatch_start:minibatch_start + minibatch_size]
            _, new_log_probabilities, entropy, new_values = agent.get_action_and_value(states[minibatch], actions[minibatch])
            ratio = (new_log_probabilities - old_log_probabilities[minibatch]).exp()   # pi_new / pi_old

            minibatch_advantages = advantages[minibatch]
            minibatch_advantages = (minibatch_advantages - minibatch_advantages.mean()) / (minibatch_advantages.std() + 1e-8)

            unclipped_objective = ratio * minibatch_advantages
            clipped_objective = torch.clamp(ratio, 1 - CLIP_COEFFICIENT, 1 + CLIP_COEFFICIENT) * minibatch_advantages
            policy_loss = -torch.min(unclipped_objective, clipped_objective).mean()
            value_loss = 0.5 * ((new_values - value_targets[minibatch]) ** 2).mean()
            entropy_bonus = entropy.mean()
            total_loss = policy_loss - ENTROPY_COEFFICIENT * entropy_bonus + VALUE_COEFFICIENT * value_loss

            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRADIENT_NORM)
            optimizer.step()

            if minibatch_start in (0, ROLLOUT_STEPS - minibatch_size):
                fraction_clipped = ((ratio - 1).abs() > CLIP_COEFFICIENT).float().mean().item()
                print(f"epoch {epoch} minibatch {minibatch_start // minibatch_size}: ratio in "
                      f"[{ratio.min().item():.4f}, {ratio.max().item():.4f}] | clipped {fraction_clipped:.0%} | "
                      f"policy loss {policy_loss.item():+.4f} | value loss {value_loss.item():.2f} | entropy {entropy_bonus.item():.4f}")


if __name__ == "__main__":
    # Self-check GAE on the hand example: rewards [1,1,1], V=[2.5,1.8,0.9], episode ends after the 3rd step
    hand_advantages = compute_gae(torch.tensor([1., 1., 1.]), torch.tensor([2.5, 1.8, 0.9]),
                                  torch.tensor([0., 0., 1.]), 0.0, 0.99, 0.95)
    assert torch.allclose(hand_advantages, torch.tensor([0.4560, 0.1850, 0.1000]), atol=1e-3)
    main()
