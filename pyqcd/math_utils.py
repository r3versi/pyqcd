import numpy as np

def tr_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Computes 1 - 1/2^n |Tr[A_dag B]|
    
    Arguments:
        a, b {np.ndarray} -- unitary matrices
    Returns:
        float -- the trace distance
    """
    return 1 - 1/(a.shape[0]) * np.abs(np.trace(np.dot(np.conjugate(a.T), b)))


def d1(a: np.ndarray, b: np.ndarray) -> float:
    return np.sum(np.abs(a - b))


def d2(a: np.ndarray, b: np.ndarray) -> float:
    return np.sqrt(np.sum(np.abs(a - b)**2))


def d_inf(a: np.ndarray, b: np.ndarray) -> float:
    return np.max(np.abs(a - b))
