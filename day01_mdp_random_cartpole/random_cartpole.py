import gymnasium as gym

NUMBER_OF_EPISODES = 10
DISCOUNT_FACTOR = 0.99
BASE_SEED = 42


def compute_discounted_return(reward_list, discount_factor):
    # Walk backwards: G_t = r_{t+1} + gamma * G_{t+1}
    discounted_return = 0.0
    for reward in reversed(reward_list):
        discounted_return = reward + discount_factor * discounted_return
    return discounted_return


def run_random_policy(number_of_episodes, discount_factor, base_seed):
    environment = gym.make("CartPole-v1")
    environment.action_space.seed(base_seed)  # makes random actions reproducible
    all_episode_returns = []

    for episode_index in range(number_of_episodes):
        observation, info = environment.reset(seed=base_seed + episode_index)
        rewards_this_episode = []
        episode_is_over = False

        while not episode_is_over:
            random_action = environment.action_space.sample()  # 0 = push left, 1 = push right
            observation, reward, terminated, truncated, info = environment.step(random_action)
            rewards_this_episode.append(reward)
            episode_is_over = terminated or truncated

        undiscounted_return = sum(rewards_this_episode)
        discounted_return = compute_discounted_return(rewards_this_episode, discount_factor)
        all_episode_returns.append(undiscounted_return)
        print(f"episode {episode_index:2d} | steps {len(rewards_this_episode):3d} | "
              f"return {undiscounted_return:5.1f} | discounted return {discounted_return:6.2f} | "
              f"ended by {'terminated' if terminated else 'truncated'}")

    environment.close()
    average_return = sum(all_episode_returns) / len(all_episode_returns)
    print(f"\naverage return over {number_of_episodes} episodes: {average_return:.1f}")
    return all_episode_returns


if __name__ == "__main__":
    assert compute_discounted_return([1, 1, 1], 0.5) == 1.75
    assert abs(compute_discounted_return([0, 0, 10], 0.9) - 8.1) < 1e-9
    run_random_policy(NUMBER_OF_EPISODES, DISCOUNT_FACTOR, BASE_SEED)
