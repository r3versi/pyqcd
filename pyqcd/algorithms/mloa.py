from .base import *
from .gloa import GLOA

class MLOA(GLOA):
    """Group Leader Optimization Algorithm
    based on papers https://arxiv.org/abs/1004.2242 and https://aip.scitation.org/doi/abs/10.1063/1.3575402
    """

    def __init__(self,
                 target: np.ndarray,
                 alphabet: Alphabet,
                 n_groups: int,
                 group_size: int,
                 circuit_size: int,
                 weights: np.ndarray = np.array([0.7, 0.15, 0.15]),
                 mat_dist: typing.Callable = tr_distance) -> None:
        """        
        Arguments:
            target {np.ndarray} -- unitary target
            alphabet {Alphabet} -- universal set alphabet
            n_groups {int} -- number of groups
            group_size {int} -- size of a group
            circuit_size {int} -- size of an individual (i.e. number of instructions)
            weights {np.ndarray} -- weights of respectively current, leader and random 
                                    in one-way crossover (default: {np.array([0.7,0.15,0.15])})
            mat_dist {typing.Callable} -- matrix distance (default: {tr_distance})
        """
        super().__init__(target, alphabet, n_groups, group_size, circuit_size, weights, mat_dist)

    def evolve(self) -> None:
        # Determine group leaders
        leaders = [min(group, key=lambda x: x.score) for group in self.groups]
        # Determine overall leader
        best = min(leaders, key=lambda x: x.score)
        # Set it as best if so
        self.update_best(best)

        # Next generation
        self.mutation()
        self.migration()
        self.refinement()
        
        self.gen += 1

    def refinement(self, n_selections: int = 10, n_iters: int = 25) -> None:
        for group in self.groups:
            for p in np.random.choice(group, size=min(n_selections, len(group)), replace=False):
            #for p in sorted(group, key=lambda x: x.score)[:min(n_selections, len(group))]:
                self.refine(p, n_iters)

        leaders = [min(group, key=lambda x: x.score) for group in self.groups]
        for leader in leaders:
            self.refine(leader, n_iters)

    def refine(self, circuit: Circuit, n_iters: int) -> Circuit:
        for _ in range(n_iters):
            new = deepcopy(circuit)

            for instr in new.instructions:
                instr.params = self.alphabet.get_random_angles(instr.n_params())

            new.score = self.fitness(new)
            if new.score < circuit.score:
                circuit.instructions = new.instructions
                circuit.score = new.score                
