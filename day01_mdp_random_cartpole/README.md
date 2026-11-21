# Day 01: MDPs and the return, with a random agent on CartPole

## The idea
Reinforcement learning is a loop. The agent sees a **state**, picks an **action**, and the environment answers
with a **reward** and the next state. That loop is a Markov decision process (MDP).

```
        action a_t
agent ────────────────► environment
  ▲                          │
  └──── state s_{t+1}, reward r_{t+1}
```

The agent's goal is the **return**, the discounted sum of future rewards:

    G_t = r_{t+1} + γ·r_{t+2} + γ²·r_{t+3} + ...  =  r_{t+1} + γ·G_{t+1}

The recursive form lets you compute every return in one pass **backwards** over an episode. `γ` (discount, 0–1)
makes near rewards count more than far ones.

CartPole: keep a pole balanced on a cart. The reward is +1 for every step it stays up. An episode is
**terminated** when the pole falls and **truncated** at 500 steps (time limit). The difference matters later,
because you should not bootstrap past a true ending.

## Run
    python day01_mdp_random_cartpole/random_cartpole.py

It checks `compute_discounted_return` against two hand-worked cases, then plays 10 seeded episodes with random actions.

## Output (real run)
```
episode  0 | steps  30 | return  30.0 | discounted return  26.03 | ended by terminated
episode  1 | steps  46 | return  46.0 | discounted return  37.02 | ended by terminated
...
average return over 10 episodes: 29.2
```
A random policy keeps the pole up for about 20–45 steps. That is the baseline every later day has to beat.

## Dry run: `compute_discounted_return([1, 1, 1], 0.5)`
| step (reversed) | reward | `discounted_return = reward + 0.5 * discounted_return` |
|---|---|---|
| start | – | 0.0 |
| last reward | 1 | 1 + 0.5·0.0 = **1.0** |
| middle | 1 | 1 + 0.5·1.0 = **1.5** |
| first | 1 | 1 + 0.5·1.5 = **1.75** |

The answer is 1.75, which equals 1 + 0.5 + 0.25. The script asserts this before it runs anything.

## Read
Sutton & Barto, *Reinforcement Learning: An Introduction*, ch. 1 and ch. 3 (free at incompleteideas.net/book).
