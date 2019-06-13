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
            qubits = np.random.choice(np.arange(self.Q), size=gate.n_qubits, replace=False)
            params = np.random.rand(gate.n_params)*2*np.pi
            out.append(Instruction(gate, qubits, params))

        return out
