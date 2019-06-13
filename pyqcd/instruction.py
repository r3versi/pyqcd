import typing
import numpy as np

from .gates import Gate

class Instruction:
    """Quantum Instruction class, stores instruction type, target qubits and optional real parameters"""
    def __init__(self,
                 gate: Gate,
                 qubits: typing.Sequence[int],
                 params: typing.Optional[typing.Sequence[float]] = None) -> None:
        """Initialize a quantum instruction
        
        Arguments:
            gate {Gate} -- Gate derived class
            qubits {typing.Sequence[int]} -- sequence of target qubits
            params {typing.Optional[typing.Sequence[float]]} -- sequence of params (default: {None})
        
        """
        self.gate = gate
        self.qubits = qubits
        self.params = params

    def to_matrix(self) -> np.ndarray:
        """Return the matrix representation of the instruction"""
        if len(self.params):
            return self.gate(*self.params).to_matrix()
        return self.gate().to_matrix()

    def __str__(self) -> str:
        return "%s, (%s), (%s)" % (self.gate.name, ",".join(str(x) for x in self.qubits), ",".join("%0.2f" % x for x in self.params))
