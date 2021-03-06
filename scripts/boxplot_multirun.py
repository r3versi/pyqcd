import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

solvers = ["GA", "GLOA", "MLOA", "MLOA2"]
results = {}

for solver in solvers:
    results[solver] = []

    files = sorted(list(Path('../data').glob('%s/*.pickle' % solver)))
    for file in files:
        with open(file, 'rb') as f:
            data = pickle.load(f)
            results[solver].append(np.min(data['best_fit']))

fig, ax = plt.subplots()
ax.set_title("QFT 2 - 100k evals - 4 runs")
ax.boxplot(results.values())
ax.set_xticklabels(solvers)
plt.show()
