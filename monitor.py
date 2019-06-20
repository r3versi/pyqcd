import pickle
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import os
from pathlib import Path

def pick_file():
    #files = sorted(glob("data/*.pickle"))
    files = sorted(list(Path('data').glob('**/*.pickle')))

    print("Pick a file to monitor")
    for x in enumerate(files):
        print("[%d] %s" % x)

    return files[int(input("\n> "))]

def run_animation(filename):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    
    def animate(i):
        try:
            with open(filename, "rb") as f:
                data = pickle.load(f)
                ax.clear()
                for label in data:
                    #HACK: show only fitness during monitor
                    if "fit" not in label:
                        continue

                    ax.plot(data[label], label=label)
                ax.legend()
        except Exception:
            pass
    
    ani = animation.FuncAnimation(fig, animate, interval=1000)
    plt.show()

def main():
    
    filename = pick_file()
    run_animation(filename)

if __name__ == "__main__":
    main()
