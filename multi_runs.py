from pyqcd.alphabet import Alphabet
from pyqcd.gates import I, U3, CX
from pyqcd.algorithms import *
from pyqcd.logger import Logger
from pyqcd.matrices import QFT

from time import time

def main():
    """Submit several runs"""

    qubits = 2
    target_name = "QFT"
    gates_set = [I, U3, CX]
    n_runs = 4

    target = QFT(qubits)
    # Instantiate a custom set of gates, an alphabet whose words are gates
    alphabet = Alphabet(Q=qubits)
    alphabet.register_gates(gates_set)

    for n_run in range(n_runs):

        start_time = int(time())

        solver = MLOA(target=target, alphabet=alphabet, n_groups=50, group_size=5, circuit_size=15)
        #solver = GA(target=target, alphabet=alphabet, pop_size=50, circuit_size=15)
        # Instantiate the logger class to keep track of fitness evolution
        
        logger = Logger("data/%s/%s_%s%d.pickle" %
                        (solver.__class__.__name__, start_time, target_name, qubits), True)
        logger.add_variables(*solver.stats().keys())

        while solver.n_evals < 100000:
            # One step evolution
            solver.evolve()

            # Gather and save statistics
            logger.register(**solver.stats())

        print("=============================")
        print("%s %s%d #%d" % (solver.__class__.__name__, target_name, qubits, n_run))
        print("Generations %d" % solver.gen)
        print("Fitness evals %d" % solver.n_evals)
        print("Score %0.5f" % solver.best.score)
        print("%s" % solver.best)
        print("=============================")

        with open("data/%s/%s_%s%d.qasm" % (solver.__class__.__name__, start_time, target_name, qubits), 'w') as f:
            f.write(solver.best.to_qasm())


if __name__ == "__main__":
    main()
