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


agent = UCB1(NUM_ARMS)

reward_history = []
arm_history = []

for t in range(NUM_USERS):
    arm = agent.select_arm()
    reward = np.random.normal(loc=true_q[arm], scale=1.0)
    agent.update(arm, reward)
    reward_history.append(reward)
    arm_history.append(arm)

reward_history = np.array(reward_history)
arm_history = np.array(arm_history)

estimated_means = agent.values
top5 = np.argsort(estimated_means)[-5:][::-1]

print("\n===== TOP 5 RECOMMENDATIONS =====")
print(f"{'Rank':<6}{'Arm':<6}{'Est. Mean':>10}{'True q(a)':>10}{'Pulls':>8}")
print("-" * 42)
for rank, a in enumerate(top5, start=1):
    print(f"{rank:<6}{a:<6}{estimated_means[a]:>+10.3f}{true_q[a]:>+10.3f}{int(agent.counts[a]):>8}")

cumulative_avg = np.cumsum(reward_history) / np.arange(1, NUM_USERS + 1)

plt.figure(figsize=(9, 4))
plt.plot(cumulative_avg, linewidth=1.5)
plt.xlabel("User interaction (t)")
plt.ylabel("Cumulative average reward")
plt.title("UCB1 — Cumulative Average Reward over 2,500 Users")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plot_cumulative_reward.png", dpi=150)
plt.show()

arm_indices = np.arange(NUM_ARMS)

plt.figure(figsize=(10, 5))
bar_width = 0.35
plt.bar(arm_indices - bar_width / 2, true_q, bar_width, label="True quality q(a)")
plt.bar(arm_indices + bar_width / 2, estimated_means, bar_width, label="Estimated mean (UCB1)")
plt.xlabel("Arm index")
plt.ylabel("Value")
plt.title("True Item Qualities vs UCB1 Estimated Means")
plt.xticks(arm_indices)
plt.legend()
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("plot_true_vs_estimated.png", dpi=150)
plt.show()
