"""Online Recommendation System using Multi-Armed Bandit (UCB1)."""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

NUM_ARMS = 15
NUM_USERS = 2500

true_q = np.random.normal(loc=0.0, scale=1.0, size=NUM_ARMS)

print("True item qualities (hidden from the algorithm):")
for i, q in enumerate(true_q):
    print(f"  Arm {i:2d}: q = {q:+.4f}")


class UCB1:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.total_pulls = 0

    def select_arm(self):
        for a in range(self.n_arms):
            if self.counts[a] == 0:
                return a

        # UCB1 exploration bonus: sqrt(2 * ln(t) / N_a)
        exploration_bonus = np.sqrt(2 * np.log(self.total_pulls) / self.counts)
        ucb_values = self.values + exploration_bonus
        return int(np.argmax(ucb_values))

    def update(self, arm, reward):
        self.total_pulls += 1
        self.counts[arm] += 1
        n = self.counts[arm]
        # Incremental sample average update
        self.values[arm] += (reward - self.values[arm]) / n
