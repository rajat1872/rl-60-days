# Day 05: PPO and GAE, one update traced number by number

## The idea
REINFORCE with a baseline (Day 4) uses each episode **once**, then throws it away. That wastes data, and one big
gradient step can wreck the policy. **PPO** (Schulman et al. 2017) reuses a rollout for several epochs, but stops
the new policy from drifting too far from the one that collected the data.

**1. Ratio.** How much more (or less) likely the new policy makes the action that was actually taken:

    r_t(θ) = π_θ(a_t|s_t) / π_old(a_t|s_t)          (= 1.0 at the start of every update)

**2. Clipped objective.** Take the more pessimistic of the plain and the clipped term, so there's no reward for
pushing r past 1 ± ε:

    L_clip = E[ min( r_t · A_t ,  clip(r_t, 1−ε, 1+ε) · A_t ) ]          ε = 0.2

```
 A_t > 0 (good action):   gain grows with r … until r = 1.2, then flat  → stop making it MORE likely
 A_t < 0 (bad action):    gain grows as r falls … until r = 0.8, then flat → stop making it LESS likely
```

**3. GAE** (Generalised Advantage Estimation) gives a lower-variance advantage than G_t − V. It's built from one-step
TD errors, discounted by γλ:

    δ_t = r_t + γ·V(s_{t+1})·(1−done_t) − V(s_t)
    A_t = δ_t + γλ·(1−done_t)·A_{t+1}            (λ = 0.95: between pure TD (λ=0) and Monte Carlo (λ=1))

**4. Full loss** = −L_clip + 0.5 · value_loss − 0.01 · entropy, with gradient-norm clipping at 0.5.

```
rollout 128 steps (no grad)  ──►  GAE advantages + value targets  ──►  4 epochs × 4 shuffled minibatches of 32
                                                                         ratio → clip → loss → Adam step
```

## Run
    python day05_ppo_gae/ppo_one_update_trace.py

This does one rollout and one PPO update on CartPole and prints the numbers you'd normally never see. It finishes
in a few seconds. The structure mirrors CleanRL's single-file `ppo.py`, with one environment instead of four.

## Output (real run, seed 1)
```
rollout: 128 steps, 7 episodes ended inside it
first 4 steps  | value [-0.108, -0.1, -0.11, -0.102] | advantage [10.17, 9.74, 9.31, 8.82] | target [10.06, 9.64, 9.2, 8.72]
epoch 0 minibatch 0: ratio in [1.0000, 1.0000] | clipped 0% | policy loss -0.0000 | value loss 35.36 | entropy 0.6901
epoch 0 minibatch 3: ratio in [0.9573, 1.0395] | clipped 0% | policy loss +0.0033 | value loss 28.82 | entropy 0.6892
epoch 1 minibatch 3: ratio in [0.9327, 1.0677] | clipped 0% | policy loss +0.0054 | value loss 27.57 | entropy 0.6900
epoch 2 minibatch 3: ratio in [0.9001, 1.0846] | clipped 0% | policy loss +0.0039 | value loss 24.92 | entropy 0.6887
epoch 3 minibatch 3: ratio in [0.8888, 1.1110] | clipped 0% | policy loss -0.0045 | value loss 29.14 | entropy 0.6886
```
What to notice:
- **The ratio starts at exactly 1.0.** The first minibatch is evaluated with the same weights that collected the
  data.
- **The ratio drifts outward with every epoch** ([0.96, 1.04] → [0.89, 1.11]). That drift is what the clip exists to
  cap. With ε = 0.2 nothing gets clipped in this first update; in a long run you'd watch `clipped %` and the
  approximate KL.
- **The untrained critic predicts about −0.1, but the targets are about 10**, so value loss starts at 35 and falls.
- **Entropy stays near ln 2 ≈ 0.693.** The policy is still almost 50/50, as it should be on update 1.

## Dry run: `compute_gae` on the hand example (asserted at the top of the script)
Inputs: rewards `[1, 1, 1]`, values `V = [2.5, 1.8, 0.9]`, and the episode ends after step 2. That means
`episode_ended_after_step = [0, 0, 1]` and `value_after_last_step = 0`, with γ = 0.99 and λ = 0.95. The loop walks
backwards:

| step | `next_state_value` | `not_done` | `delta = r + γ·V_next·not_done − V` | `running_advantage = delta + γλ·not_done·previous` |
|---|---|---|---|---|
| 2 | 0.0 (after last step) | 0 | 1 + 0 − 0.9 = **0.100** | 0.100 + 0 = **0.100** |
| 1 | V[2] = 0.9 | 1 | 1 + 0.99·0.9 − 1.8 = **0.091** | 0.091 + 0.9405·0.100 = **0.185** |
| 0 | V[1] = 1.8 | 1 | 1 + 0.99·1.8 − 2.5 = **0.282** | 0.282 + 0.9405·0.185 = **0.456** |

So `advantages = [0.456, 0.185, 0.100]`, and the value targets are `advantages + values = [2.956, 1.985, 1.000]`.
`not_done = 0` at step 2 is the important bit. It stops the advantage of one episode leaking into the next one in the
same rollout. Get that wrong and the critic learns from returns that never happened.

## Read
- Schulman et al. 2017, *Proximal Policy Optimization Algorithms* (arXiv:1707.06347), §3 and §5.
- Schulman et al. 2015, *High-Dimensional Continuous Control Using GAE* (arXiv:1506.02438), §3.
- Huang et al., "The 37 Implementation Details of PPO" (ICLR Blog Track 2022).
