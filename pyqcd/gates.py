import typing

import numpy as np

from pyqcd import matrices


"""
Matrices assume big endian basis ordering. So that ket{q0}*ket{q1} is represented as ket{q1 q0}.
2-qubit basis order: {00, 01, 10, 11}
Matrix repr. of CX 0 1 (where 0 is control and 1 target):
1 0 0 0
0 0 0 1
0 0 1 0
0 1 0 0
"""


class Gate(object):
    def __init__(self, name: str, n_qubits: int, params: typing.Optional[typing.Sequence[float]] = None) -> None:
        self.name = name

        self.n_qubits = n_qubits
        self.n_params = 0

        self.params = params

        if self.params is not None:
            self.n_params = len(params)

    def to_qasm(self) -> str:
        if self.params:
            return "%s(%s)" % (self.name, ",".join([str(x) for x in self.params]))

        return self.name

    def to_matrix(self) -> np.ndarray:
        raise NotImplementedError

    def __str__(self) -> str:
        return self.to_qasm()


class I(Gate):
    name = "id"
    n_qubits = 1
    n_params = 0

    def __init__(self) -> None:
        super().__init__("id", 1, [])

    def to_matrix(self) -> np.ndarray:
        return matrices.I


class X(Gate):
    name = "x"
    n_qubits = 1
    n_params = 0

    def __init__(self) -> None:
        super().__init__("x", 1, [])

    def to_matrix(self) -> np.ndarray:
        return matrices.X


class Y(Gate):
    name = "y"
    n_qubits = 1
    n_params = 0

    def __init__(self) -> None:
        super().__init__("y", 1, [])

    def to_matrix(self) -> np.ndarray:
        return matrices.Y


class Z(Gate):
    name = "z"
    n_qubits = 1
    n_params = 0

    def __init__(self) -> None:
        super().__init__("z", 1, [])

    def to_matrix(self) -> np.ndarray:
        return matrices.Z


class RX(Gate):
    name = "rx"
    n_qubits = 1
    n_params = 1

    def __init__(self, a: float):
        super().__init__("rx", 1, [a])

    def to_matrix(self) -> np.ndarray:
        a = float(self.params[0])
        return np.array(
            [
                [np.cos(a/2), -1j*np.sin(a/2)],
                [-1j*np.sin(a/2), np.cos(a/2)]
            ], dtype=complex)


class RY(Gate):
    name = "ry"
    n_qubits = 1
    n_params = 1

    def __init__(self, a: float):
        super().__init__("ry", 1, [a])

    def to_matrix(self) -> np.ndarray:
        a = float(self.params[0])
        return np.array(
            [
                [np.cos(a/2), -np.sin(a/2)],
                [np.sin(a/2), np.cos(a/2)]
            ], dtype=complex)


class RZ(Gate):
    name = "rz"
    n_qubits = 1
    n_params = 1

    def __init__(self, a: float):
        super().__init__("rz", 1, [a])

    def to_matrix(self) -> np.ndarray:
        a = float(self.params[0])
        return np.array(
            [
                [np.exp(-1j*a/2), 0],
                [0, np.exp(1j*a/2)]
            ], dtype=complex)


class H(Gate):
    name = "h"
    n_qubits = 1
    n_params = 0

    def __init__(self) -> None:
        super().__init__("h", 1, [])

    def to_matrix(self) -> np.ndarray:
        return matrices.H


class T(Gate):
    name = "t"
    n_qubits = 1
    n_params = 0

    def __init__(self) -> None:
        super().__init__("t", 1, [])

    def to_matrix(self) -> np.ndarray:
        return matrices.T


class Tdg(Gate):
    name = "tdg"
    n_qubits = 1
    n_params = 0

    def __init__(self) -> None:
        super().__init__("tdg", 1, [])

    def to_matrix(self) -> np.ndarray:
        return matrices.Tdg


class S(Gate):
    name = "s"
    n_qubits = 1
    n_params = 0

    def __init__(self) -> None:
        super().__init__("s", 1, [])

    def to_matrix(self) -> np.ndarray:
        return matrices.S


class Sdg(Gate):
    name = "sdg"
    n_qubits = 1
    n_params = 0

    def __init__(self) -> None:
        super().__init__("sdg", 1, [])

    def to_matrix(self) -> np.ndarray:
        return matrices.Sdg


class V(Gate):
    name = "v"
    n_qubits = 1
    n_params = 0

    def __init__(self) -> None:
        super().__init__("v", 1, [])

    def to_matrix(self) -> np.ndarray:
        return matrices.V


class Vdg(Gate):
    name = "vdg"
    n_qubits = 1
    n_params = 0

    def __init__(self) -> None:
        super().__init__("vdg", 1, [])

    def to_matrix(self) -> np.ndarray:
        return matrices.Vdg


class U1(Gate):
    name = "u1"
    n_qubits = 1
    n_params = 1

    def __init__(self, a: float):
        super().__init__("u1", 1, [a])

    def to_matrix(self) -> np.ndarray:
        a = float(self.params[0])
        return np.array(
            [
                [1, 0],
                [0, np.exp(1j * a)]
            ], dtype=complex)


class U2(Gate):
    name = "u2"
    n_qubits = 1
    n_params = 2

    def __init__(self, a: float, b: float):
        super().__init__("u2", 1, [a, b])

    def to_matrix(self) -> np.ndarray:
        a, b = [float(x) for x in self.params]
        return 1/np.sqrt(2)*np.array(
            [
                [1,                 -np.exp(1j * b)],
                [np.exp(1j * a),    np.exp(1j * (a + b))]
            ], dtype=complex)


class U3(Gate):
    name = "u3"
    n_qubits = 1
    n_params = 3

    def __init__(self, a: float, b: float, c: float):
        super().__init__("u3", 1, [a, b, c])

    def to_matrix(self) -> np.ndarray:
        a, b, c = [float(x) for x in self.params]
        return np.array(
            [
                [np.cos(a / 2),                    -
                 np.exp(1j * c) * np.sin(a / 2)],
                [np.exp(1j * b) * np.sin(a / 2),
                 np.exp(1j * (b + c)) * np.cos(a / 2)]
            ], dtype=complex)


class CX(Gate):
    name = "cx"
    n_qubits = 2
    n_params = 0

    def __init__(self) -> None:
        super().__init__("cx", 2, [])

    def to_matrix(self) -> np.ndarray:
        return matrices.CX


class CZ(Gate):
    name = "cz"
    n_qubits = 2
    n_params = 0

    def __init__(self) -> None:
        super().__init__("cz", 2, [])

    def to_matrix(self) -> np.ndarray:
        return matrices.CZ


class CCX(Gate):
    name = "ccx"
    n_qubits = 3
    n_params = 0

    def __init__(self) -> None:
        super().__init__("ccx", 3, [])

    def to_matrix(self) -> np.ndarray:
        return matrices.CCX
