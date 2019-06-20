from .base import *

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


    def stats(self) -> typing.Dict:
        res = {}
        res['best_fit'] = self.best.score if self.best is not None else None

        for idx, group in enumerate(self.groups):
            res["mean_fit_%d"%idx] = np.mean([x.score for x in group])
        res['n_evals'] = self.n_evals
        return res

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
        self.gen += 1

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

    def migration(self) -> None:
        """Perform one-way-crossover: unidirectional migration between different groups"""
        t = np.random.randint(3*self.group_size*self.circuit_size//2 + 1)
        for i in range(self.n_groups):
            for _ in range(t):
                x = np.random.randint(self.n_groups)
                k = np.random.randint(self.group_size)
                pr = np.random.randint(self.circuit_size)

                new = deepcopy(self.groups[i][k])
                new.instructions[pr] = deepcopy(
                    self.groups[x][k].instructions[pr])
                new.score = self.fitness(new)

                if new.score < self.groups[i][k].score:
                    self.groups[i][k] = new

    def combine(self, current: Circuit, leader: Circuit, random: Circuit) -> Circuit:
        """Generate a new circuit combining current, leader and random"""
        instr = []
        for ops in zip(current.instructions, leader.instructions, random.instructions):
            if ops[0] == ops[1] and ops[0] == ops[2] and ops[0].n_params:
                #Combine parameters arithmetically
                params = np.array([x.params for x in ops])
                new_params = np.sum((params.T * self.weights).T, axis=0)

                instr.append(ops[0], np.random.choice(
                    [x.qubits for x in ops], p=self.weights))
            else:
                instr.append(np.random.choice(ops, p=self.weights))

        if len(instr) < len(current.instructions):
            instr.extend(current.instructions[len(instr):])

        return Circuit(self.Q, instr)
