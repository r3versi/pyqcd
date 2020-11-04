from .base import *
from .gloa import GLOA


class MLOA(GLOA):
    """Memetic Group Leader Optimization Algorithm
    based on papers https://arxiv.org/abs/1004.2242 and https://aip.scitation.org/doi/abs/10.1063/1.3575402
    GLOA + real parameters refinement + smart migration
    """

    def __init__(self,
                 target: np.ndarray,
                 alphabet: Alphabet,
                 n_groups: int,
                 group_size: int,
                 circuit_size: int,
                 weights: np.ndarray = np.array([0.7, 0.15, 0.15]),
                 ref_pb: float = 0.25,
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
        super().__init__(target, alphabet, n_groups,
                         group_size, circuit_size, weights, mat_dist)

        self.ref_pb = ref_pb
        # Extra stats initialization
        self.n_refs = 0

    def evolve(self) -> None:
        # Next generation
        self.mutation()
        self.migration()
        self.refinement()
        self.gen += 1

        # Determine group leaders
        leaders = [min(group, key=lambda x: x.score) for group in self.groups]
        # Determine overall leader
        best = min(leaders, key=lambda x: x.score)
        # Set it as best if so
        self.update_best(best)

    def stats(self) -> typing.Dict:
        res = super().stats()
        res['n_refs'] = self.n_refs
        return res

    def migration(self) -> None:
        # Compute origin group probabilities: the lower the fitness, the better
        group_p = [1/np.mean([x.score for x in group]) for group in self.groups]
        group_p = group_p/np.sum(group_p)

        t = np.random.randint(3*self.group_size*self.circuit_size//2 + 1)

        for _ in range(self.n_groups):
            gid = np.random.choice(self.n_groups, p=group_p)
            for _ in range(t):
                i = np.random.randint(self.n_groups)
                j = np.random.randint(self.group_size)
                k = np.random.randint(self.circuit_size)

                new = self.groups[i][j].clone()
                new.instructions[k] = self.groups[gid][j].instructions[k].clone()
                new.score = self.fitness(new)

                if new.score < self.groups[i][j].score:
                    self.groups[i][j] = new
                    self.n_migs += 1

    def refinement(self, n_selections: int = 1, n_iters: int = 10) -> None:
        for group in self.groups:
            for p in np.random.choice(group, size=min(n_selections, len(group)), replace=False):
                self.refine(p, n_iters)

        leaders = [min(group, key=lambda x: x.score) for group in self.groups]
        for leader in leaders:
            self.refine(leader, n_iters)

    def refine(self, circuit: Circuit, n_iters: int) -> Circuit:
        for _ in range(n_iters):
            new = circuit.clone()

            for instr in new.instructions:
                if instr.n_params() and np.random.rand() < self.ref_pb:
                    instr.params = instr.params + \
                        self.alphabet.get_random_angles(instr.n_params())/4
                    new.score = None

            if new.score is None:
                new.score = self.fitness(new)

            if new.score < circuit.score:
                circuit.instructions = new.instructions
                circuit.score = new.score
                self.n_refs += 1

        return circuit
