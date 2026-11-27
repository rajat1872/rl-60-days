# Day 08: The RLHF pipeline: a reward model from preferences, and the KL penalty

## The idea
RLHF (InstructGPT, Ouyang et al. 2022) has three stages:

```
 1. SFT                 2. REWARD MODEL                         3. RL (PPO)
 base model ──► fine-  humans compare two answers:              maximise   E[ r_φ(x, y) ]  −  β · KL( π_θ || π_SFT )
 tune on demos         "A is better than B"                                 ▲                    ▲
        │              ──► train r_φ so r(A) > r(B)                 reward model          stay close to the
        └──────────────────────────────────────────────────────►    score                 model you started from
```

**Bradley–Terry** turns "A beat B" into something you can train on:

    P(A ≻ B) = σ( r(A) − r(B) )            loss = − log σ( r(chosen) − r(rejected) )

Only **differences** of reward matter. Add a constant to every score and nothing changes, so the reward model has no
natural scale.

**The KL penalty** keeps the policy near the reference (SFT) model. Without it, RL finds whatever the reward model
over-scores (Day 12). For a fixed set of answers the KL-regularised objective has a closed form:

    π*(y) ∝ π_ref(y) · exp( r(y) / β )

`β` is the knob. A large β stays near the reference; a small β chases the reward model.

## Run
    python day08_rlhf_reward_model_and_kl/reward_model_and_kl.py
NumPy only; it takes about a second.

## Output (real run, seed 0)
```
== 1. reward model from 2000 preference pairs ==
true weights    {'helpfulness': 2.0, 'length': 0.0, 'toxicity': -3.0}
learned weights {'helpfulness': 2.04, 'length': 0.04, 'toxicity': -2.98}
held-out pairwise accuracy 0.900 (the true reward itself scores 0.899: labels are noisy)

== 2. KL-regularised policy over 5 candidate responses ==
reward-model scores: [ 2.04  0.41  2.24 -0.04  1.42]
  beta                               policy  E[reward]     KL
 10.00           [0.43 0.18 0.11 0.17 0.1 ]       1.34   0.00
  1.00           [0.63 0.06 0.19 0.04 0.08]       1.85   0.25
  0.30           [0.66 0.   0.32 0.   0.02]       2.09   0.66
  0.05           [0.07 0.   0.93 0.   0.  ]       2.23   1.95
```
- **The reward model recovered the hidden preference** from nothing but "A beat B" labels: about +2 helpfulness,
  −3 toxicity, and length about 0.
- **Its held-out accuracy (0.900) matches the true reward's own (0.899).** The labels are noisy, so 90% is the
  ceiling. You can't beat it by training harder.
- **As β falls, expected reward rises but KL grows fast.** At β = 0.05 the policy puts 93% on the long, slightly
  toxic answer #3, because the reward model scores it 0.2 higher. That's a real, if small, preference the reward
  model learned. In Day 12, the gap between "reward model score" and "what people actually want" is exactly what a
  low β exploits.

## Dry run 1: `bradley_terry_loss([2.0], [1.0])` (asserted in the script)
1. `margins = chosen_rewards − rejected_rewards = [1.0]`
2. `np.log1p(np.exp(−1.0))` = ln(1 + 0.3679) = **0.3133**. That equals −ln σ(1), because σ(1) = 0.7311 and
   −ln 0.7311 = 0.3133.
3. With equal rewards (margin 0), the loss is ln 2 = 0.6931: the model is 50/50 and learns the most from this pair.
   A huge positive margin drives the loss toward 0, so pairs the model already gets right barely move it.

## Dry run 2: `kl_regularised_optimum(model_rewards, reference_policy, beta=1.0)`
Inputs: `model_rewards = [2.04, 0.41, 2.24, −0.04, 1.42]` and `reference_policy = [0.4, 0.2, 0.1, 0.2, 0.1]`.

1. `unnormalised = reference · exp(rewards / 1.0)` = `[0.4·7.69, 0.2·1.51, 0.1·9.39, 0.2·0.96, 0.1·4.14]` =
   `[3.076, 0.301, 0.939, 0.192, 0.414]`
2. Their sum is **4.923**, so `policy = unnormalised / 4.923` = **`[0.625, 0.061, 0.191, 0.039, 0.084]`**. That
   matches the β = 1.00 row up to rounding.
3. Answer #0 started at 0.4 and rises to 0.63, because it's high-reward *and* the reference already liked it.
   Answer #2 scores highest but started at 0.1, so at β = 1 it only reaches 0.19. The reference holds it back.
   Shrink β and that hold weakens: at β = 0.05 it reaches 0.93.

## Read
- Ouyang et al. 2022, *Training language models to follow instructions with human feedback* (InstructGPT),
  arXiv:2203.02155, §3.
- Nathan Lambert, *RLHF Book* (rlhfbook.com): the reward modelling and regularisation chapters.

*Rebuilt Nov 27: the original Day 8 script was lost when the workspace was wiped. This is a fresh version of the same
lesson, run before publishing.*
