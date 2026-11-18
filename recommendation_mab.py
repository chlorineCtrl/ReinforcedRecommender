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
