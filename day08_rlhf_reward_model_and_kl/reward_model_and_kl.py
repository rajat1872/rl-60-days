"""RLHF in miniature: (1) learn a reward model from pairwise preferences (Bradley-Terry),
(2) see what the KL penalty does when you optimise a policy against that reward.

Responses are described by 3 features: [helpfulness, length, toxicity]. The HIDDEN human preference is
true_reward = 2*helpfulness - 3*toxicity (length doesn't matter). The reward model only ever sees which of two
responses a labeller preferred, never the true reward.
"""
import numpy as np

TRUE_WEIGHTS = np.array([2.0, 0.0, -3.0])          # helpfulness, length, toxicity
FEATURE_NAMES = ["helpfulness", "length", "toxicity"]


def sigmoid(value):
    return 1.0 / (1.0 + np.exp(-value))


def bradley_terry_loss(chosen_rewards, rejected_rewards):
    # P(chosen beats rejected) = sigmoid(r_chosen - r_rejected); maximise its log  ->  loss = -log sigmoid(margin)
    margins = chosen_rewards - rejected_rewards
    return float(np.mean(np.log1p(np.exp(-margins))))


def make_preference_pairs(number_of_pairs, random_generator):
    first, second = random_generator.normal(size=(number_of_pairs, 3)), random_generator.normal(size=(number_of_pairs, 3))
    # Labellers are noisy: they prefer the truly better response with probability sigmoid(true margin)
    first_preferred = random_generator.random(number_of_pairs) < sigmoid(first @ TRUE_WEIGHTS - second @ TRUE_WEIGHTS)
    chosen = np.where(first_preferred[:, None], first, second)
    rejected = np.where(first_preferred[:, None], second, first)
    return chosen, rejected


def train_reward_model(chosen, rejected, learning_rate=0.5, steps=500):
    # Linear reward model r(x) = w . x, trained by gradient descent on the Bradley-Terry loss
    weights = np.zeros(3)
    feature_differences = chosen - rejected
    for _ in range(steps):
        margins = feature_differences @ weights
        gradient = -(feature_differences * (1 - sigmoid(margins))[:, None]).mean(axis=0)   # d loss / d w
        weights -= learning_rate * gradient
    return weights


def kl_divergence(policy, reference):
    return float(np.sum(policy * np.log(policy / reference)))


def kl_regularised_optimum(rewards, reference, beta):
    # argmax_pi  E_pi[r] - beta * KL(pi || reference)  has the closed form  pi* ∝ reference * exp(r / beta)
    unnormalised = reference * np.exp(rewards / beta)
    return unnormalised / unnormalised.sum()


def main():
    random_generator = np.random.default_rng(0)

    print("== 1. reward model from 2000 preference pairs ==")
    chosen, rejected = make_preference_pairs(2000, random_generator)
    learned_weights = train_reward_model(chosen, rejected)
    print("true weights   ", {name: round(float(value), 2) for name, value in zip(FEATURE_NAMES, TRUE_WEIGHTS)})
    print("learned weights", {name: round(float(value), 2) for name, value in zip(FEATURE_NAMES, learned_weights)})
    test_chosen, test_rejected = make_preference_pairs(2000, random_generator)
    held_out_accuracy = np.mean(test_chosen @ learned_weights > test_rejected @ learned_weights)
    best_possible = np.mean(test_chosen @ TRUE_WEIGHTS > test_rejected @ TRUE_WEIGHTS)
    print(f"held-out pairwise accuracy {held_out_accuracy:.3f} (the true reward itself scores {best_possible:.3f}: "
          f"labels are noisy)")

    print("\n== 2. KL-regularised policy over 5 candidate responses ==")
    candidates = np.array([[1.0, 0.0, 0.0],     # helpful, normal length, clean
                           [0.2, 0.0, 0.0],     # unhelpful
                           [1.2, 2.0, 0.1],     # very helpful, very long, slightly toxic
                           [0.0, -1.0, 0.0],    # terse
                           [2.0, 0.5, 0.9]])    # most "helpful" but toxic
    reference_policy = np.array([0.4, 0.2, 0.1, 0.2, 0.1])          # what the SFT model already does
    model_rewards = candidates @ learned_weights
    print("reward-model scores:", model_rewards.round(2))
    print(f"{'beta':>6s} {'policy':>36s} {'E[reward]':>10s} {'KL':>6s}")
    for beta in (10.0, 1.0, 0.3, 0.05):
        policy = kl_regularised_optimum(model_rewards, reference_policy, beta)
        print(f"{beta:6.2f} {np.array2string(policy.round(2)):>36s} {policy @ model_rewards:10.2f} "
              f"{kl_divergence(policy, reference_policy):6.2f}")
    print("small beta -> chases the reward model, drifts far from the reference (big KL); "
          "large beta -> stays near the reference")


if __name__ == "__main__":
    # Hand check: margin 1 -> loss = log(1 + e^-1) = 0.3133;  margin 0 -> log 2 = 0.6931
    assert abs(bradley_terry_loss(np.array([2.0]), np.array([1.0])) - 0.3133) < 1e-4
    assert abs(bradley_terry_loss(np.array([1.0]), np.array([1.0])) - 0.6931) < 1e-4
    # Hand check: KL([0.5, 0.5] || [0.25, 0.75]) = 0.5 ln 2 + 0.5 ln(2/3) = 0.1438
    assert abs(kl_divergence(np.array([0.5, 0.5]), np.array([0.25, 0.75])) - 0.1438) < 1e-4
    main()
