from .base import *
import sympy as sp

from ..math_utils import sympy_tr_distance

from scipy.optimize import basinhopping

class SympyBasinHopping(BaseSearch):

    def __init__(self,
                 target: np.ndarray,
                 mat_dist: typing.Callable = sympy_tr_distance) -> None:
        
        self.target = target
        self.mat_dist = mat_dist

    def optimize(self, circuit: Circuit) -> None:

        parametrized = circuit.get_sympy_copy()

        score = sympy_tr_distance(self.target, parametrized.to_matrix(True))
        params = parametrized.get_params()

        #grad = [sp.diff(score, x) for x in params]
        func = sp.lambdify(params, score)

        x0 = circuit.get_params()
        bounds = np.array([[0, 2*np.pi] for _ in range(len(x0))])
        minimizer_kwargs = {"method": "L-BFGS-B", "bounds": bounds}

        print("Continuous optimization of %d params" % len(x0))
        print("Input circuit score: ", circuit.score)
        ret = basinhopping(
            func, x0, minimizer_kwargs=minimizer_kwargs, niter=20, niter_success=5)

        print("In the end\n", ret.x, "\nscore", ret.fun)

        circuit.set_parameters(ret.x)
        circuit.score = ret.fun
        return circuit


class BasinHopping(BaseSearch):
    """
    Real parameters optimizer.
    """

    def __init__(self,
                 target: np.ndarray,
                 circuit: Circuit,
                 mat_dist: typing.Callable = tr_distance) -> None:
        """        
        Arguments:
            target {np.ndarray} -- unitary target
            mat_dist {typing.Callable} -- matrix distance (default: {tr_distance})
        """

        self.target = target
        self.circuit = circuit
        self.mat_dist = mat_dist
    
    def run(self):
        def f(x):
            circuit = self.set_parameters(x)
            E = self.fitness(circuit)
            return E 

        x0 = self.circuit.get_params()

        bounds = np.array([[0, 2*np.pi] for _ in range(len(x0))])
        minimizer_kwargs = {"method": "L-BFGS-B", "bounds": bounds}

        print("Input circuit score: ", self.circuit.score)
        ret = basinhopping(f, x0, minimizer_kwargs=minimizer_kwargs, niter=20, niter_success=5)

        print("In the end\n", ret.x, "\nscore", ret.fun)

        circuit = self.set_parameters(ret.x)
        circuit.score = ret.fun
        return circuit

    def set_parameters(self, params: np.ndarray) -> Circuit:
        
        circuit = deepcopy(self.circuit)
        k = 0
        for idx, instr in enumerate(circuit.instructions):
            for i in range(instr.gate.n_params):
                circuit.instructions[idx].params[i] = params[k]
                k += 1
        return circuit
