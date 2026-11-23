# Day 03: Policy gradients and REINFORCE on CartPole

## The idea
Day 1's agent acted at random. Today the policy is a small neural network, π(a|s) = "probability of pushing left or
right given the cart's 4 numbers", and it learns from its own episodes.

**REINFORCE** in one sentence: *make the actions that came before high returns more likely, and the ones before low
returns less likely.* The policy-gradient theorem turns that into a formula you can differentiate:

    ∇J(θ) = E[ Σ_t  G_t · ∇ log π_θ(a_t | s_t) ]

```
   play one episode with the current policy (sample actions, don't argmax)
        │   states, actions, log π(a_t|s_t), rewards
        ▼
   compute G_t for every step, backwards (Day 1's recursion)
        │
        ▼
   loss = − Σ_t G_t · log π(a_t|s_t)      ← minimising this = gradient ASCENT on expected return
        │
        ▼
   one Adam step on θ, then repeat
```

Why `log π`? The **log-derivative trick** gives ∇π = π · ∇log π. That turns "gradient of an expectation" into an
expectation of something you can sample by just playing. You never need to differentiate the environment.

## Run
    python day03_policy_gradients_reinforce/reinforce_cartpole.py

It runs 1000 episodes and takes about 75 s on a laptop CPU. The CPU build of torch is enough:
`pip install torch --index-url https://download.pytorch.org/whl/cpu`.

## Output (real run, seed 0)
```
episodes    1-  50 | average return   24.6
episodes  101- 150 | average return   46.1
episodes  201- 250 | average return  109.4
episodes  301- 350 | average return  197.1
episodes  351- 400 | average return  154.1
episodes  451- 500 | average return  315.8
episodes  601- 650 | average return  392.0
episodes  701- 750 | average return  184.5
episodes  801- 850 | average return  391.6
episodes  951-1000 | average return  308.1
```
It learns: from about 25 steps (random) to 300–390. But look how it **swings** (392 → 184 → 391). Plain REINFORCE
has high variance, because every action in an episode is pushed by the *whole* noisy return. Day 4 fixes this with a
baseline.

## Dry run: `compute_returns_for_every_step([0, 0, 10], 0.9)`
The script asserts this before training:

| reversed step | reward | `running_return = reward + 0.9 · running_return` | appended |
|---|---|---|---|
| start | – | 0.0 | – |
| t = 2 | 10 | 10 + 0.9·0 = **10.0** | [10.0] |
| t = 1 | 0 | 0 + 0.9·10 = **9.0** | [10.0, 9.0] |
| t = 0 | 0 | 0 + 0.9·9 = **8.1** | [10.0, 9.0, 8.1] |

After `reverse()` you get `[8.1, 9.0, 10.0]`. The first action gets credit 8.1 for a reward that came two steps later.

**One update of `main()`:**
1. `run_one_episode` resets the env, and at each step:
   - the network gives a `Categorical` over {left, right};
   - `sample()` picks an action;
   - `log_prob(action)` is stored, which keeps the gradient path to θ;
   - the env steps.
2. Say the episode lasts 20 steps. Then `rewards = [1.0] × 20`, and `returns` runs from G_0 ≈ 18.2 down to G_19 = 1.0.
3. `loss = −(log_probabilities * returns).sum()`. Early actions get the biggest weight, because more reward followed
   them.
4. `loss.backward()` fills `.grad` on every weight; `optimizer.step()` moves θ to make those 20 actions more likely,
   in proportion to their returns.

## Read
Sutton & Barto ch. 13 (§13.1–13.3). For the full derivation, see OpenAI Spinning Up, "Intro to Policy Optimization".
