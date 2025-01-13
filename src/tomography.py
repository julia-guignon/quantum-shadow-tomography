import numpy as np

from src.qu_utils import *
from src.circuit_logic import *

def multi_qubits_lin_tomography(n, valid=True, nshots=1000, noise=False, expectation_values=None, measurements=None, simulator=None, verbose=True):
    if expectation_values==None or measurements==None:
        expectation_values, measurements = measurement_process(n, nshots, noise, simulator, verbose=verbose)

    rhotilde = contruct_rho_lin(n, expectation_values, measurements, valid)
    return rhotilde

        
def contruct_rho_lin(n, expect_val, measurements, valid):
    rhotilde = np.sum([scalar*mat for scalar, mat in zip(list(expect_val.values()), list(map(get_pauli_str_mat, measurements.keys())))], axis=0)/2**n # /2**n for the normalization of states
    if valid:
        # return the nearest positiv semi-definite matrix
        return make_it_semi_positive_definite(rhotilde.todense())
    else:
        return rhotilde.todense()



def multi_qubits_ML_tomography(n, method='Powell', maxit=200, nshots=1000, eps=1e-6, noise=False, expectation_values=None, measurements=None):
    if expectation_values==None or measurements==None:
        expectation_values, measurements = measurement_process(n, nshots, noise) #[1]+[0]*(4**n-1)
    rhotilde = contruct_rho_lin(n, expectation_values, measurements, valid=True)
    # L, _ = sy.linalg.lu(rhotilde, permute_l=True)
    # L = sy.linalg.cholesky(rhotilde, lower=True)
    L = np.linalg.cholesky(rhotilde+eps*np.eye(2**n))
    L = L/np.trace(L)
    # L = Cholesky_Decomposition(rhotilde)
    print('Cholesky done')
    optimization = sy.optimize.minimize(loss_function, get_t_from_mat(L), method=method, args=(expectation_values, measurements))#, options={'maxiter':maxit})
    opt_t = optimization.x
    print('Optimization done')
    T = parametrization_state_ML(opt_t)
    TdaggerT = T.transpose().conj()@T
    rhotilde = TdaggerT/TdaggerT.trace()
    return rhotilde.todense()