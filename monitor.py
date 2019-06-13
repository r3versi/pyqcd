import pickle
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from glob import glob
import os

def pick_file():
    files = sorted(glob("data/*.pickle"))
    print("Pick a file to monitor")
    for x in enumerate(files):
        print("[%d] %s" % x)

    return files[int(input())]

def monitor():
    
    file = pick_file()
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    def animate(i):
        try:
            with open(file, "rb") as f:
                data = pickle.load(f)
                ax.clear()
                for label in data:
                    ax.plot(data[label], label=label)
                ax.legend()
        except Exception:
            pass

    ani = animation.FuncAnimation(fig, animate, interval=1000)
    plt.show()

if __name__ == "__main__":
    monitor()
