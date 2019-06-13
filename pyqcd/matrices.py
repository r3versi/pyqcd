import numpy as np 

I = np.eye(2, dtype=complex)

X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

H = 1/np.sqrt(2)*np.array([[1, 1], [1, -1]], dtype=complex)

S = np.array([[1, 0], [0, 1j]], dtype=complex)
Sdg = np.array([[1, 0], [0, -1j]], dtype=complex)

T = np.array([[1, 0], [0, np.exp(np.pi*0.25j)]], dtype=complex)
Tdg = np.array([[1, 0], [0, np.exp(-np.pi*0.25j)]], dtype=complex)

V = 1/2 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=complex)
Vdg = 1/2 * np.array([[1 - 1j, 1 + 1j], [1 + 1j, 1 - 1j]], dtype=complex)

CX = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
CZ = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]], dtype=complex)

CCX = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 1, 0, 0, 0, 0]], dtype=complex)
Toffoli = CCX

def Identity(Q: int) -> np.ndarray:
    return np.eye(2**Q)

def QFT(Q: int) -> np.ndarray:
    w = np.exp(2j*np.pi/(2**Q))
    mat = np.array([[w**(i*j) for j in range(2**Q)]
                for i in range(2**Q)], dtype=complex)/np.sqrt(2**Q)
    return mat