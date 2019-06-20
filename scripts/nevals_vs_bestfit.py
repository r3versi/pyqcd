import pickle
import numpy as np
import matplotlib.pyplot as plt

import os
from pathlib import Path

solvers = ["GA", "GLOA", "MLOA"]
colors = ["blue", "orange", "red"]

fig, ax = plt.subplots()

for color,solver in zip(colors,solvers):
    files = sorted(list(Path('../data').glob('%s/*.pickle' % solver)))
    for file in files:
        with open(file, 'rb') as f:
            data = pickle.load(f)
            x = data['n_evals']
            y = data['best_fit']

            ax.plot(x,y,label=solver, color=color)

ax.set_title("QFT 2 - 100k evals - 4 runs")
ax.legend()
ax.set_xlabel("Fitness evals")
ax.set_ylabel("Best fitness")
plt.show()
