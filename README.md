# 60 days of RL for agents

One day of a 60-day plan toward RL-environment and agent-harness engineering, published every day: the concept,
the code, and real output from running it. It starts at MDPs and ends at building and red-teaming training
environments for coding agents.

## Setup
    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt

Each `dayNN_*/` folder has a README (the idea, how to run it, what the output means) and runnable code.
Code uses long descriptive names and asserts on hand-checked values, so every script tests itself when it runs.

## Log
| day | topic | run |
|---|---|---|
| 01 | MDPs, returns, discounting; random policy on CartPole | `python day01_mdp_random_cartpole/random_cartpole.py` |
| 02 | Multi-armed bandits, epsilon-greedy, regret | `python day02_bandits_epsilon_greedy/epsilon_greedy_bandit.py` |
| 03 | Policy gradients, the log-derivative trick, REINFORCE | `python day03_policy_gradients_reinforce/reinforce_cartpole.py` |
| 04 | Baselines, advantage, actor-critic (3 seeds, plain vs baseline) | `python day04_baselines_advantage/reinforce_with_baseline.py baseline 0` |
| 05 | PPO clipped objective + GAE, one update traced | `python day05_ppo_gae/ppo_one_update_trace.py` |
| 06 | Gymnasium API, terminated vs truncated, custom text env ConfigFix-v0 | `python day06_gymnasium_custom_env/test_and_agents.py` |
| 08 | RLHF pipeline: Bradley–Terry reward model, KL-regularised policy | `python day08_rlhf_reward_model_and_kl/reward_model_and_kl.py` |
| 09 | DPO: preference loss on the policy, Z cancels, implicit reward | `python day09_dpo/dpo_tabular.py` |
