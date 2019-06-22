from .base import *


class MC(BaseSearch):
    """Vanilla MC search"""

    def __init__(self,
                 target: np.ndarray,
                 alphabet: Alphabet,
                 circuit_size: int,
                 mat_dist: typing.Callable = tr_distance) -> None:
        """
        Arguments:
            target {np.ndarray} -- unitary target
            alphabet {Alphabet} -- universal set alphabet
            circuit_size {int} -- size of an individual (i.e. number of instructions)
            mat_dist {typing.Callable} -- matrix distance (default: {tr_distance})
        """
        super().__init__(target, alphabet, circuit_size, mat_dist)

    def stats(self) -> typing.Dict:
        res = super().stats()
        return res

    def evolve(self) -> None:
        new = self.get_random_circuit()
        new.score = self.fitness(new)

        self.update_best(new)
        self.gen += 1
