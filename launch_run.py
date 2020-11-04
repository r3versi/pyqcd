import signal
import sys
from time import time

from pyqcd.algorithms import GA, GLOA, MC, MLOA
from pyqcd.alphabet import Alphabet
from pyqcd.gates import CX, U3, I
from pyqcd.logger import Logger
from pyqcd.matrices import QFT, random_unitary


def main():
    start = time()
    # Quantum Fourier Transform on 4 qubits
    target = random_unitary(Q=4)

    # Instantiate a custom set of gates, an alphabet whose words are gates
    alphabet = Alphabet(Q=4)
    alphabet.register_gates([I, U3, CX])

    # Instantiate the search class
    solver = MLOA(target=target, alphabet=alphabet,
                  n_groups=5, group_size=5, circuit_size=50)

    # Instantiate the logger class to keep track of fitness evolution
    logger = Logger("data/%s_%srandom.pickle" %
                    (int(time()), solver.__class__.__name__), True)
    logger.add_variables(*solver.stats().keys())

    def exit_handler(sig, frame):
        solver.end()
        sys.exit(0)

    signal.signal(signal.SIGINT, exit_handler)

    # Main loop
    while solver.n_evals < 500000:
        # One step evolution
        solver.evolve()

        # Gather and save statistics
        logger.register(**solver.stats())

    solver.end()
    print("Time elapsed %d s" % (time() - start))


if __name__ == "__main__":
    main()
