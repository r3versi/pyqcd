from time import time

from pyqcd.algorithms import GA, GLOA, MC, MLOA
from pyqcd.alphabet import Alphabet
from pyqcd.gates import CX, U3, I
from pyqcd.logger import Logger
from pyqcd.math_utils import d2
from pyqcd.matrices import QFT


def main():
    # Quantum Fourier Transform on 4 qubits
    target = QFT(Q=2)

    # Instantiate a custom set of gates, an alphabet whose words are gates
    alphabet = Alphabet(Q=2)
    alphabet.register_gates([I, U3, CX])

    # Instantiate the search class
    solver = MLOA(target=target, alphabet=alphabet,
                  n_groups=2, group_size=1, circuit_size=15)
    #solver = GA(target=target, alphabet=alphabet, pop_size=50, circuit_size=15)
    #solver = MC(target=target, alphabet=alphabet, circuit_size=15)

    # Instantiate the logger class to keep track of fitness evolution
    logger = Logger("data/%s_%s_QFT2.pickle" %
                    (int(time()), solver.__class__.__name__), True)
    logger.add_variables(*solver.stats().keys())

    # Main loop
    while solver.best is None or solver.best.score > 0.01:
        # One step evolution
        solver.evolve()

        # Gather and save statistics
        logger.register(**solver.stats())

    print("=============================")
    print("Generations %d" % solver.gen)
    print("Fitness evals %d" % solver.n_evals)
    print("Score %0.2f" % solver.best.score)
    print("%s" % solver.best)
    print("=============================")

    with open('best.qasm', 'w') as f:
        f.write(solver.best.to_qasm())


if __name__ == "__main__":
    main()
