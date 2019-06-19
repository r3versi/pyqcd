import typing
import numpy as np

from .gates import Gate
from .instruction import Instruction

class Alphabet(object):
    """An object to represent a set S of quantum instructions acting on Q qubits"""

    def __init__(self, Q: int) -> None:
        """Initialize an Alphabet, a collection of gates
        
        Arguments:
            Q {int} -- number of qubits        
        """
        self.Q = Q
        self.gates: typing.List[Gate] = []

    def register_gates(self, gates: typing.Sequence[Gate]):
        """Registers a gate
        
        Arguments:
            gate {Gate} -- an object derived from class Gate
        """
        for gate in gates:
            if gate not in self.gates:
                self.gates.append(gate)

    def get_random_qubits(self, n: int) -> typing.Sequence[int]:
        return np.random.choice(np.arange(self.Q), size=n, replace=False)

    def get_random_angles(self, n: int) -> typing.Sequence[float]:
        return np.random.rand(n)*2*np.pi

    def get_random(self, n: int = 1) -> typing.List[Instruction]:
        """Get a random instruction from register

        Arguments:
            n {int} -- number of instructions (default: {1})

        Returns:
            typing.List[Instruction] -- list of instructions
        """
        out = []
        for _ in range(n):
            gate = np.random.choice(self.gates)
            qubits = self.get_random_qubits(gate.n_qubits)
            params = self.get_random_angles(gate.n_params)
            out.append(Instruction(gate, qubits, params))

        return out
