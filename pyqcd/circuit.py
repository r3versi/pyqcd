import typing
import numpy as np
import sympy as sp
from copy import deepcopy

from qiskit import QuantumCircuit
from qiskit.providers.basicaer.basicaertools import einsum_matmul_index

from .alphabet import Instruction
from .utils.object_einsum import object_einsum


class UnitaryCircuit(object):
    """Unitary representation of a circuit"""

    def __init__(self, Q: int, to_sympy: bool = False) -> None:
        """Initialize a unitary circuit
        
        Arguments:
            Q {int} -- number of qubits
        """
        self.Q = Q
        if to_sympy:
            self.dtype = object
        else:
            self.dtype = complex

        self._unitary = np.reshape(np.eye(2**Q, dtype=self.dtype), Q * [2, 2])

    def add_one_qubit(self, 
                      gate: typing.Union[np.ndarray, sp.Matrix], 
                      qubit: int) -> None:
        """Append a 1-qubit gate

        Arguments:
            gate {np.ndarray} -- matrix representation of the gate
            qubit {int} -- target qubit
        """
        gate_tensor = np.array(gate, dtype=self.dtype)
        indexes = einsum_matmul_index([qubit], self.Q)
        #self._unitary = np.einsum(
        #    indexes, gate_tensor, self._unitary, dtype=complex)
        self._unitary = object_einsum(indexes, gate_tensor, self._unitary)

    def add_two_qubits(self, 
                       gate: typing.Union[np.ndarray, sp.Matrix],
                       qubit0: int,
                       qubit1: int) -> None:
        """Append a 2-qubit gate
        
        Arguments:
            gate {np.ndarray} -- matrix representation of the gate
            qubit0 {int} -- first target qubit
            qubit1 {int} -- second target qubit
        """
        gate_tensor = np.reshape(np.array(gate, dtype=self.dtype), 4 * [2])
        indexes = einsum_matmul_index([qubit0, qubit1], self.Q)
        #self._unitary = np.einsum(
        #    indexes, gate_tensor, self._unitary, dtype=complex, casting='no')
        self._unitary = object_einsum(indexes, gate_tensor, self._unitary)

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

    def get_sympy_copy(self) -> "Circuit":
        """Returns a circuit with the same instructions, but real valued parameters substituted by Sympy symbols
        
        Returns:
            Circuit -- parametrized circuit
        """
        new = Circuit(self.Q, [])
        new.score = None

        nparams = self.get_nparams()
        params = sp.symbols('p0:%d' % nparams)
        k = 0
        for instr in self.instructions:
            new_instr = deepcopy(instr)
            # cast to object to store sympy symbols
            if instr.gate.n_params:
                new_instr.params = new_instr.params.astype(object)
                for i in range(new_instr.gate.n_params):
                    new_instr.params[i] = params[k]
                    k+= 1
            new.instructions.append(new_instr)
        
        return new

    def get_nparams(self) -> int:
        n_params = sum([x.gate.n_params for x in self.instructions])
        return n_params
    
    def get_params(self) -> np.ndarray:
        params = []
        for instr in self.circuit.instructions:
            if instr.gate.n_params:
                params.extend(instr.params)
        return np.array(params)

    def set_params(self, params: np.ndarray) -> None:
        k = 0
        for idx, instr in enumerate(self.instructions):
            for i in range(instr.gate.n_params):
                self.instructions[idx].params[i] = params[k]
                k += 1

    def __len__(self) -> int:
        return len(self.instructions)

    def to_matrix(self, to_sympy: bool = False) -> np.ndarray:
        """Return the matrix representation of the circuit"""
        UC = UnitaryCircuit(self.Q, to_sympy)
        for i in self.instructions:
            if i.gate.n_qubits == 1:
                UC.add_one_qubit(i.to_matrix(to_sympy), *i.qubits)
            else:
                UC.add_two_qubits(i.to_matrix(to_sympy), *i.qubits)
        
        return UC.to_matrix()

    def to_qiskit_circuit(self) -> QuantumCircuit:
        qasm_str = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[%d];\n" % self.Q

        for i in self.instructions:
            qasm_str += "%s" % i.gate.name
            if i.gate.n_params:
                qasm_str += "(%s) " % (','.join("%0.2f" %p for p in i.params))
            qasm_str += " %s;\n" % (','.join("q[%d]" % q for q in i.qubits))

        return QuantumCircuit.from_qasm_str(qasm_str)

    def __str__(self) -> str:
        return str(self.to_qiskit_circuit())
