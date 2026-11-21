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
