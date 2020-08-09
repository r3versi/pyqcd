import typing
import numpy as np

from pyqcd.circuit import Circuit
from pyqcd.alphabet import Alphabet
from pyqcd.math_utils import tr_distance


class BaseSearch:
    """Common Base for search algorithms"""

    def __init__(self,
                 target: np.ndarray,
                 alphabet: Alphabet,
                 circuit_size: int,
                 mat_dist: typing.Callable = tr_distance) -> None:
        """
        Initialize BaseSearch.

        Arguments:
            target {np.ndarray} -- unitary target
            alphabet {Alphabet} -- universal set alphabet
            mat_dist {typing.Callable} -- matrix distance
                                          (default: {tr_distance})
        """
        self.Q = int(np.log2(target.shape[0]))
        self.target = target
        self.alphabet = alphabet
        self.circuit_size = circuit_size
        self.mat_dist = mat_dist

        self.best = None
        self.gen = 0
        self.n_evals = 0

    def stats(self) -> typing.Dict:
        """Return current stats

        Returns:
            typing.Dict -- dict whose keys are monitored values
        """
        res = {}
        res['best_fit'] = self.best.score if self.best is not None else None
        res['n_evals'] = self.n_evals
        return res

    def matrix_distance(self, circuit: Circuit) -> float:
        """Distance between circuit and self.target
        as defined by self.mat_dist

        Arguments:
            circuit {Circuit} -- a circuit obj

        Returns:
            float -- the distance
        """
        return self.mat_dist(circuit.to_matrix(), self.target)

    def circuit_cost(self, circuit: Circuit) -> float:
        """Implementation cost of circuit

        Arguments:
            circuit {Circuit} -- a circuit obj

        Returns:
            float -- the cost to implement a circuit
        """
        return 0

    def fitness(self, circuit: Circuit) -> float:
        """Return total fitness: distance + cost

        Arguments:
            circuit {Circuit} -- a circuit obj

        Returns:
            float -- fitness
        """
        self.n_evals += 1
        return self.matrix_distance(circuit) + self.circuit_cost(circuit)

    def get_random_circuit(self) -> Circuit:
        """Return a random circuit

        Returns:
            Circuit -- a circuit obj
        """
        return Circuit(self.Q, self.alphabet.get_random(self.circuit_size))

    def update_best(self, circuit: Circuit) -> None:
        """Update current best if circuit is better

        Arguments:
            circuit {Circuit} -- a circuit obj
        """
        if self.best is None or circuit.score < self.best.score:
            self.best = circuit.clone()
            # print("New best @ gen %d, score %0.5f\n%s" %
            #      (self.gen, self.best.score, self.best))

    def end(self) -> None:

        print("=============================")
        print("Generations %d" % self.gen)
        print("Fitness evals %d" % self.n_evals)
        print("Score %0.2f" % self.best.score)
        print("%s" % self.best)
        print("=============================")

        with open('best.qasm', 'w') as f:
            f.write(self.best.to_qasm())
