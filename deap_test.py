from pyqcd.circuit import UnitaryCircuit, Circuit
from pyqcd.instruction import Instruction
from pyqcd.gates import I, RX, RY, RZ, CX
from pyqcd.math_utils import tr_distance

import numpy as np

from deap import algorithms, tools, base, creator


CX_PB = 0.25
MUT_PB = 0.75
N_GEN = 100
N_QUBITS = 5
IND_SIZE = 100
POP_SIZE = 50

target = np.identity(2**N_QUBITS)

target = np.load("data/benchmark_circuits/random0_5q.npy")


class Experiment:
    """Experiment is a collection of parameters to run """

    def __init__(self, name: str, Q: int, circuit_size: int, target: np.ndarray) -> None:
        """Initialize an experiment

        Args:
            name (str): name of the experiment
            Q (int): number of qubits
            circuit_size (int): maximum number of gates in a circuit
            target (np.ndarray): unitary matrix representing target circuit
        """

        self.Q = Q
        self.circuit_size = circuit_size
        self.target = target


experiments = [
    Experiment("id_5q", 5, 20, np.identity(2**5))
]


def generate_dictionary(qubits):
    dictionary = {}

    i = 0
    # ID gates
    for q in range(qubits):
        dictionary[i] = Instruction(I, [q], [])
        i += 1

    # Rx, Ry, Rz
    for q in range(qubits):
        for a in range(32):
            angle = a*np.pi/8
            dictionary[i] = Instruction(RX, [q], [angle])
            dictionary[i+1] = Instruction(RY, [q], [angle])
            dictionary[i+2] = Instruction(RZ, [q], [angle])
            i += 3

    # CX gates
    for q in range(qubits):
        for p in range(q+1, qubits):
            dictionary[i] = Instruction(CX, [q, p], [])
            i += 1

    return dictionary


###################


instructions = generate_dictionary(N_QUBITS)
n_instructions = len(instructions)


def evaluateInd(target, individual):
    matrix = Circuit(N_QUBITS, [instructions[i]
                                for i in individual]).to_matrix()

    return tr_distance(target, matrix),


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("gene", np.random.randint, 0, n_instructions)
toolbox.register("individual", tools.initRepeat,
                 creator.Individual, toolbox.gene, n=IND_SIZE)
toolbox.register("population", tools.initRepeat,
                 list, toolbox.individual, n=POP_SIZE)

toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate", tools.mutUniformInt, low=0,
                 up=n_instructions-1, indpb=MUT_PB)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluateInd, target)

hof = tools.HallOfFame(1)

population = toolbox.population()
population, logbook = algorithms.eaSimple(
    population, toolbox, CX_PB, MUT_PB, N_GEN, halloffame=hof)

for individual in hof:
    print(id(individual))
    print(individual)
    print(individual.fitness)

for individual in population:
    print(id(individual))
    print(individual)
    print(individual.fitness)
