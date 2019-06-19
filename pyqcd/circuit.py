import typing
import numpy as np

from qiskit import QuantumCircuit
from qiskit.providers.basicaer.basicaertools import einsum_matmul_index

from .alphabet import Instruction


class UnitaryCircuit(object):
    """Unitary representation of a circuit"""

    def __init__(self, Q: int) -> None:
        """Initialize a unitary circuit
        
        Arguments:
            Q {int} -- number of qubits
        """
        self.Q = Q
        self._unitary = np.reshape(np.eye(2**Q, dtype=complex), Q * [2, 2])

    def add_one_qubit(self, gate: np.ndarray, qubit: int) -> None:
        """Append a 1-qubit gate

        Arguments:
            gate {np.ndarray} -- matrix representation of the gate
            qubit {int} -- target qubit
        """
        gate_tensor = np.array(gate, dtype=complex)
        indexes = einsum_matmul_index([qubit], self.Q)
        self._unitary = np.einsum(
            indexes, gate_tensor, self._unitary, dtype=complex, casting='no')

    def add_two_qubits(self, gate: np.ndarray, qubit0: int, qubit1: int) -> None:
        """Append a 2-qubit gate
        
        Arguments:
            gate {np.ndarray} -- matrix representation of the gate
            qubit0 {int} -- first target qubit
            qubit1 {int} -- second target qubit
        """
        gate_tensor = np.reshape(np.array(gate, dtype=complex), 4 * [2])
        indexes = einsum_matmul_index([qubit0, qubit1], self.Q)
        self._unitary = np.einsum(
            indexes, gate_tensor, self._unitary, dtype=complex, casting='no')

    def to_matrix(self) -> np.ndarray:
        """Matrix representation of the circuit
        
        Returns:
            np.ndarray -- (2**Q,2**Q) unitary matrix
        """
        return np.reshape(self._unitary, 2 * [2**self.Q])


class Circuit(object):
    """Quantum circuit as a sequence of quantum instructions"""
    def __init__(self, Q: int, instructions: typing.Sequence[Instruction]) -> None:
        """Initialize a quantum circuit
        
        Arguments:
            Q {int} -- number of qubits
            instructions {typing.Sequence[Instruction]} -- sequence of quantum instructions
        """
        self.Q = Q
        self.score = None
        self.instructions = instructions

    def __len__(self) -> int:
        return len(self.instructions)

    def to_matrix(self) -> np.ndarray:
        """Return the matrix representation of the circuit"""
        UC = UnitaryCircuit(self.Q)
        for i in self.instructions:
            if i.gate.n_qubits == 1:
                UC.add_one_qubit(i.to_matrix(), *i.qubits)
            else:
                UC.add_two_qubits(i.to_matrix(), *i.qubits)
        return UC.to_matrix()

    def to_qasm(self) -> str:
        """Return circuit as QASM string"""
        qasm_str = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[%d];\n" % self.Q

        for i in self.instructions:
            qasm_str += "%s" % i.gate.name
            if i.gate.n_params:
                qasm_str += "(%s) " % (','.join("%0.2f" % p for p in i.params))
            qasm_str += " %s;\n" % (','.join("q[%d]" % q for q in i.qubits))
        
        return qasm_str

    def to_qiskit_circuit(self) -> QuantumCircuit:
        """Return qiskit.QuantumCircuit"""
        return QuantumCircuit.from_qasm_str(self.to_qasm())

    def __str__(self) -> str:
        return str(self.to_qiskit_circuit())
