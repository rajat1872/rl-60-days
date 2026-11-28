# Day 09: DPO, preference optimisation without a reward model or an RL loop

## The idea
Day 8's RLHF needs two models and an RL loop: train a reward model, then run PPO with a KL penalty. **DPO** (Rafailov
et al. 2023) notices that the KL-regularised objective has a closed-form optimum:

    π*(y|x) = π_ref(y|x) · exp( r(x,y) / β ) / Z(x)

Solve that for the reward:

    r(x,y) = β · log( π*(y|x) / π_ref(y|x) ) + β · log Z(x)

Then substitute it into Bradley–Terry. A preference only looks at the **difference** r(x, y_w) − r(x, y_l), and the
**β·log Z(x) term is identical for both answers, so it cancels.** That matters, because Z sums over every possible
response and can't be computed. What's left is a loss on the policy itself:

    L_DPO = − log σ( β · [ log π_θ(y_w)/π_ref(y_w)  −  log π_θ(y_l)/π_ref(y_l) ] )

```
 RLHF:  preference pairs ──► reward model r_φ ──► PPO on π_θ with −β·KL(π_θ‖π_ref) ──► policy
 DPO:   preference pairs ──────────────────────────► one supervised loss on π_θ (π_ref frozen) ──► policy
                         the policy IS the reward model:  r̂(y) = β · log π_θ(y)/π_ref(y)
```

## Run
    python day09_dpo/dpo_tabular.py
NumPy only; it takes under a second.

The setup is one prompt and 5 candidate answers, with a "tabular" policy that has one logit per answer. 5000 noisy
preference pairs are labelled from a hidden reward `[1, 0, 2, −1, 0.5]`. β = 0.5.

## Output (real run, seed 0)
```
reference policy       [0.4 0.2 0.1 0.2 0.1]
DPO policy (5000 pairs) [0.304 0.024 0.642 0.002 0.028]
RLHF optimum (Day 8)   [0.332 0.022 0.612 0.003 0.03 ]  <- uses the TRUE reward, which DPO never saw

implicit reward beta*log(pi/ref), shifted so response 0 = true reward 1.0:
  implicit [ 1.    0.07  2.07 -1.11  0.5 ]
  true     [ 1.   0.   2.  -1.   0.5]

largest gap between DPO policy and RLHF optimum: 0.030
```
- **DPO lands within 0.03 of the policy RLHF would reach with the *true* reward,** even though it only ever saw
  pairs. The gap is label noise from 5000 noisy pairs.
- **The implicit reward β·log(π/π_ref) recovers the hidden reward up to a constant** (2.07 vs 2.0, −1.11 vs −1.0).
  That's the "your policy is secretly a reward model" result from the paper's title.
- **Response 2 started at only 10% under the reference and ends at 64%.** Response 3 is pushed to 0.2%, but never
  to exactly 0. log π must stay finite, which is the KL leash at work.

## Dry run 1: `dpo_loss(-1.0, -1.2, -2.0, -1.8, beta=0.1)` (asserted in the script)
1. Chosen log-ratio = −1.0 − (−1.2) = **+0.2**: the policy already likes the winner a bit more than the reference
   does.
2. Rejected log-ratio = −2.0 − (−1.8) = **−0.2**: and likes the loser a bit less.
3. `margin = 0.1 × (0.2 − (−0.2)) = 0.04`
4. `log1p(exp(−0.04))` = ln(1 + 0.9608) = **0.6733**. That's barely below ln 2 = 0.6931, the loss when the policy still
   equals the reference.

## Dry run 2: the first gradient step of `train_dpo` (real counts)
At the start, `logits = log(reference)`, so every log-ratio is 0, every margin is 0, `sigmoid(0) = 0.5`, and every
pair gets `weight = (1 − 0.5) · β = 0.25`.

Win and loss counts over the 5000 pairs:

| response | wins | losses | `gradient = 0.25·(losses − wins)/5000` | new logit (`lr 5`) |
|---|---|---|---|---|
| 0 | 1245 | 782 | −0.0232 | −0.916 + 0.116 = −0.801 |
| 1 | 763 | 1186 | +0.0212 | −1.715 |
| 2 | 1728 | 307 | **−0.0710** | −2.303 + 0.355 = **−1.947** |
| 3 | 272 | 1682 | **+0.0705** | −1.962 |
| 4 | 992 | 1043 | +0.0026 | −2.315 |

After one step the policy is `[0.444, 0.178, 0.141, 0.139, 0.098]`. Answer 2 (most wins) rises from 0.10 and answer
3 (most losses) falls from 0.20.

In the gradient, the `−π` terms from ∂log π/∂logits cancel between winner and loser. That's the code-level twin of
Z cancelling. As margins grow, `1 − sigmoid(margin)` shrinks, so pairs the policy already ranks correctly stop
pushing. It's self-limiting, unlike plain likelihood training.

## Read
Rafailov et al. 2023, *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*,
arXiv:2305.18290, §4 (the derivation) and §5.

*Rebuilt Nov 27: the original Day 9 script was lost when the workspace was wiped. This is a fresh version of the same
lesson, run before publishing.*
