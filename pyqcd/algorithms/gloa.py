from .base import *

from pyqcd.circuit import Circuit
from pyqcd.instruction import Instruction


class GLOA(BaseSearch):
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
        super().__init__(target, alphabet, circuit_size, mat_dist)

        self.weights = weights
        self.n_groups = n_groups
        self.group_size = group_size

        self.groups = [[self.get_random_circuit()
                        for _ in range(self.group_size)] for _ in range(self.n_groups)]
        self.compute_fitness()

        # Determine group leaders
        leaders = [min(group, key=lambda x: x.score) for group in self.groups]
        # Determine overall leader
        best = min(leaders, key=lambda x: x.score)
        # Set it as best if so
        self.update_best(best)

        # Extra stats initialization
        self.n_muts = 0
        self.n_migs = 0

    def stats(self) -> typing.Dict:
        res = super().stats()

        for idx, group in enumerate(self.groups):
            res["mean_fit_%d" % idx] = np.mean([x.score for x in group])

        res['n_muts'] = self.n_muts
        res['n_migs'] = self.n_migs
        return res

    def evolve(self) -> None:
        # Next generation
        self.mutation()
        self.migration()
        self.gen += 1

        # Determine group leaders
        leaders = [min(group, key=lambda x: x.score) for group in self.groups]
        # Determine overall leader
        best = min(leaders, key=lambda x: x.score)
        # Set it as best if so
        self.update_best(best)

    def compute_fitness(self) -> None:
        for group in self.groups:
            for p in group:
                if p.score is None:
                    p.score = self.fitness(p)

    def mutation(self) -> None:
        """Perform mutation and recombination between members of the same group"""
        for gid, group in enumerate(self.groups):
            leader = min(group, key=lambda x: x.score)

            for idx, p in enumerate(group):
                random = self.get_random_circuit()

                new_circuit = self.combine(p, leader, random)
                new_circuit.score = self.fitness(new_circuit)

                if new_circuit.score < p.score:
                    self.groups[gid][idx] = new_circuit
                    self.n_muts += 1

    def migration(self) -> None:
        """Perform one-way-crossover: unidirectional migration between different groups"""
        t = np.random.randint(3*self.group_size*self.circuit_size//2 + 1)
        for x in range(self.n_groups):
            # Migrate t genes towards group x
            for _ in range(t):
                # Get gene k from individual j of group i
                i = np.random.randint(self.n_groups)
                j = np.random.randint(self.group_size)
                k = np.random.randint(self.circuit_size)

                # Clone receiver and substitute an instruction
                new = self.groups[x][j].clone()
                new.instructions[k] = self.groups[i][j].instructions[k].clone()
                new.score = self.fitness(new)

                # Substitute the individual if new is fittest
                if new.score < self.groups[x][j].score:
                    self.groups[x][j] = new
                    self.n_migs += 1

    def combine(self, current: Circuit, leader: Circuit, random: Circuit) -> Circuit:
        """Generate a new circuit combining current, leader and random"""
        p = Circuit(self.Q, [])
        for instr in zip(current.instructions, leader.instructions, random.instructions):

            if instr[0].n_params() > 0 and instr[0].gate == instr[1].gate and instr[0].gate == instr[2].gate:
                # Combine parameters arithmetically
                params = np.array([x.params for x in instr])
                new_params = np.sum((params.T * self.weights).T, axis=0)
                new_qubits = instr[np.random.choice(
                    np.arange(3), p=self.weights)].qubits

                new_instr = Instruction(instr[0].gate, new_qubits, new_params)
                p.append(new_instr)
            else:
                new_instr = np.random.choice(instr, p=self.weights).clone()
                p.append(new_instr)

        while len(p) < len(current.instructions):
            p.append(current.instructions[len(p)].clone())

        return p
