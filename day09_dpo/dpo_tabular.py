"""DPO from scratch on one prompt with 5 candidate responses (a "tabular" policy: one logit per response).

Day 8 solved   max E[r] - beta*KL(pi || pi_ref)   =>   pi*(y) ∝ pi_ref(y) * exp(r(y) / beta).
DPO flips it around: r(y) = beta * log(pi*(y) / pi_ref(y)) + beta * log Z, then plugs that into the Bradley-Terry
preference loss. Z is the same for both answers of a pair, so it cancels, and you train the POLICY directly on
preference pairs: no reward model, no RL loop.
"""
import numpy as np

TRUE_REWARDS = np.array([1.0, 0.0, 2.0, -1.0, 0.5])         # hidden; the learner only sees preference pairs
REFERENCE_POLICY = np.array([0.4, 0.2, 0.1, 0.2, 0.1])
BETA = 0.5


def sigmoid(value):
    return 1.0 / (1.0 + np.exp(-value))


def dpo_loss(policy_logprob_chosen, reference_logprob_chosen, policy_logprob_rejected, reference_logprob_rejected, beta):
    # -log sigmoid( beta * [ (log pi(y_w) - log ref(y_w)) - (log pi(y_l) - log ref(y_l)) ] )
    margin = beta * ((policy_logprob_chosen - reference_logprob_chosen) - (policy_logprob_rejected - reference_logprob_rejected))
    return float(np.log1p(np.exp(-margin)))


def softmax(logits):
    shifted = np.exp(logits - logits.max())
    return shifted / shifted.sum()


def make_preference_pairs(number_of_pairs, random_generator):
    pairs = []
    for _ in range(number_of_pairs):
        first, second = random_generator.choice(len(TRUE_REWARDS), size=2, replace=False)
        first_wins = random_generator.random() < sigmoid(TRUE_REWARDS[first] - TRUE_REWARDS[second])   # noisy labeller
        pairs.append((first, second) if first_wins else (second, first))
    return pairs


def train_dpo(pairs, learning_rate=5.0, epochs=3000):
    logits = np.log(REFERENCE_POLICY).copy()            # start AT the reference policy
    chosen = np.array([winner for winner, _ in pairs])
    rejected = np.array([loser for _, loser in pairs])
    for _ in range(epochs):
        log_policy = np.log(softmax(logits))
        log_ratios = log_policy - np.log(REFERENCE_POLICY)
        margins = BETA * (log_ratios[chosen] - log_ratios[rejected])
        # d loss / d logits for one pair = -(1 - sigmoid(margin)) * beta * (onehot(chosen) - onehot(rejected));
        # the "- pi" terms from d log pi / d logits cancel between chosen and rejected, just like Z does
        weights = (1 - sigmoid(margins)) * BETA
        gradient = np.zeros_like(logits)
        np.add.at(gradient, chosen, -weights)
        np.add.at(gradient, rejected, weights)
        logits -= learning_rate * gradient / len(pairs)
    return softmax(logits)


def main():
    random_generator = np.random.default_rng(0)
    pairs = make_preference_pairs(5000, random_generator)
    dpo_policy = train_dpo(pairs)
    rlhf_optimum = REFERENCE_POLICY * np.exp(TRUE_REWARDS / BETA)
    rlhf_optimum /= rlhf_optimum.sum()
    print("reference policy      ", REFERENCE_POLICY.round(3))
    print("DPO policy (5000 pairs)", dpo_policy.round(3))
    print("RLHF optimum (Day 8)  ", rlhf_optimum.round(3), " <- uses the TRUE reward, which DPO never saw")
    implicit_rewards = BETA * np.log(dpo_policy / REFERENCE_POLICY)
    print("\nimplicit reward beta*log(pi/ref), shifted so response 0 = true reward 1.0:")
    print("  implicit", (implicit_rewards - implicit_rewards[0] + TRUE_REWARDS[0]).round(2))
    print("  true    ", TRUE_REWARDS)
    largest_gap = np.abs(dpo_policy - rlhf_optimum).max()
    print(f"\nlargest gap between DPO policy and RLHF optimum: {largest_gap:.3f}")


if __name__ == "__main__":
    # Hand check (beta 0.1): chosen ratio -1.0 - (-1.2) = 0.2, rejected ratio -2.0 - (-1.8) = -0.2
    # margin = 0.1 * 0.4 = 0.04 -> loss = log(1 + e^-0.04) = 0.6733
    assert abs(dpo_loss(-1.0, -1.2, -2.0, -1.8, beta=0.1) - 0.6733) < 1e-4
    # At the start (policy == reference) every margin is 0, so the loss is log 2 for every pair
    assert abs(dpo_loss(-1.0, -1.0, -2.0, -2.0, beta=0.1) - np.log(2)) < 1e-12
    main()
