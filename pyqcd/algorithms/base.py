import typing
import numpy as np

from pyqcd.circuit import Circuit
from pyqcd.alphabet import Alphabet
from pyqcd.math_utils import tr_distance


class BaseSearch:
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

    def matrix_distance(self, circuit: Circuit) -> float:
        return self.mat_dist(circuit.to_matrix(), self.target)

    def circuit_cost(self, circuit: Circuit) -> float:
        return 0

    def fitness(self, circuit: Circuit) -> float:
        self.n_evals += 1
        return self.matrix_distance(circuit) + self.circuit_cost(circuit)

    def get_random_circuit(self) -> Circuit:
        return Circuit(self.Q, self.alphabet.get_random(self.circuit_size))

    def update_best(self, circuit: Circuit) -> None:
        if self.best is None or circuit.score < self.best.score:
            self.best = circuit.clone()
            # print("New best @ gen %d, score %0.5f\n%s" %
            #      (self.gen, self.best.score, self.best))
