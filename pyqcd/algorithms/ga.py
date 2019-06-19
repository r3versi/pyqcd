from .base import *

class GA(BaseSearch):
    """Genetic Algorithm"""
    def __init__(self,
                 target: np.ndarray,
                 alphabet: Alphabet,
                 pop_size: int,
                 circuit_size: int,
                 cx_pb: float = 0.7,
                 mut_pb: float = 0.15,
                 mat_dist: typing.Callable = tr_distance) -> None:
        """
        Arguments:
            target {np.ndarray} -- unitary target
            alphabet {Alphabet} -- universal set alphabet
            pop_size {int} -- number of individuals
            circuit_size {int} -- size of an individual (i.e. number of instructions)
            cx_pb {float} -- probability of crossover (default: 0.7)
            mut_pb {float} -- probability of mutation (default: 0.15)
            mat_dist {typing.Callable} -- matrix distance (default: {tr_distance})
        """
        super().__init__(target, alphabet, circuit_size, mat_dist)

        self.cx_pb = cx_pb
        self.mut_pb = mut_pb

        self.pop_size = pop_size

        self.pop = [self.get_random_circuit() for _ in range(self.pop_size)]
        self.compute_fitness()

    def stats(self) -> typing.Dict:
        res = {}
        res['best_fit'] = self.best.score if self.best is not None else None
        res['mean_fit'] = np.mean([x.score for x in self.pop])
        #res['mean_len'] = np.mean([len(x) for x in self.pop])
        return res

    def evolve(self) -> None:
        best = min(self.pop, key=lambda x: x.score)
        self.update_best(best)

        self.fixing()
        self.new_generation()

        self.gen += 1

    def new_generation(self) -> None:
        """One evolution step"""
        np.random.shuffle(self.pop)

        for idx, (p0, p1) in enumerate(zip(self.pop[::2], self.pop[1::2])):
            c0, c1 = deepcopy(p0), deepcopy(p1)

            if np.random.rand() < self.cx_pb:
                self.mate(c0, c1)

            if np.random.rand() < self.mut_pb:
                self.mutate(c0)

            if np.random.rand() < self.mut_pb:
                self.mutate(c1)

            c0.score = self.fitness(c0)
            c1.score = self.fitness(c1)

            # Applying elitism during selection
            if c0.score <= p0.score:
                self.pop[idx] = c0
            if c1.score <= p1.score:
                self.pop[idx+1] = c1

    def fixing(self) -> None:
        """Substitute empty individuals with a new random one"""
        for idx, p in enumerate(self.pop):
            if len(p) == 0:
                self.pop[idx] = self.get_random_circuit()
                self.pop[idx].score = self.fitness(self.pop[idx])

    def compute_fitness(self) -> None:
        """Compute fitness for all individuals not yet scored"""
        for p in self.pop:
            if p.score is None:
                p.score = self.fitness(p)

    def mutate(self, p: Circuit) -> Circuit:
        """Single point mutation: a random instruction is removed, changed or added"""
        mode = np.random.randint(3)

        if mode is 0:
            # DELETE random instruction
            idx = np.random.randint(len(p))
            p.instructions.pop(idx)
        elif mode is 1:
            # CHANGE random instruction idx
            idx = np.random.randint(len(p))

            # Mutation mode
            mut_mode = np.random.randint(3)

            if mut_mode is 0:
                # NEW instruction
                p.instructions[idx] = self.alphabet.get_random()[0]
            elif mut_mode is 1:
                # QUBITS mutation
                qubits = self.alphabet.get_random_qubits(p.instructions[idx].gate.n_qubits)
                p.instructions[idx].qubits = qubits
            else:
                # PARAMS mutation
                params = self.alphabet.get_random_angles(p.instructions[idx].gate.n_params)
                p.instructions[idx].params = params
        else:
            # INSERT random instruction in random position
            idx = np.random.randint(len(p)+1)
            p.instructions.insert(idx, self.alphabet.get_random()[0])

        return p

    def mate(self, p0: Circuit, p1: Circuit) -> typing.Tuple[Circuit]:
        """Perform one point crossover"""
        idx = np.random.randint(min(len(p0), len(p1)))

        p0.instructions[:idx], p1.instructions[:idx] = p1.instructions[:idx], p0.instructions[:idx]

        return p0, p1

    def roulette_selection(self, n: int = 1) -> Circuit:
        #BUG: yields incorrect selection for minimizing problems
        scores = [x.score for x in self.pop]
        return np.random.choice(self.pop, size=n, p=scores/np.sum(scores))

    def tournament_selection(self, t_size: int) -> Circuit:
        selection = np.random.choice(self.pop, size=t_size, replace=False)
        return min(selection, key=lambda x: x.score)
