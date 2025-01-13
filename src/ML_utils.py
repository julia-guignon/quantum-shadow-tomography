import scipy as sy
import numpy as np
from src.qu_utils import *

from math import log


def get_t_from_mat(mat):
    if isinstance(mat, sy.sparse.csr_matrix):
        mat = mat.todense()

    n = int(log(mat.shape[0])/log(2))
    t = np.zeros(4**n)
    t[:2**n] = np.real(np.diag(mat))
    count = 2**n
    for i in range(1, 2**n):
        incr = 2*(2**n-i)
        t[count:count+incr:2] = np.real(np.diag(mat, -i))
        t[count+1:count+incr+1:2] = np.imag(np.diag(mat, -i))
        count += incr
    return t


def loss_function(t, expect_val_measured, measurements):
    rho_t = parametrization_state_ML(t)
    rho_t_expectation_value = lambda x: get_expectation_value_from_str(rho_t, x)
    expectation_values_parametrized = np.array(list(map(rho_t_expectation_value, measurements.keys())))
    expect_val_measured = np.array(list(expect_val_measured.values()))
    return np.sum((expectation_values_parametrized-expect_val_measured)**2)#/expect_val_measured)#expectation_values_parametrized)



def parametrization_state_ML(t):
    n = int(np.log(len(t))/np.log(4))
    matrix = sy.sparse.diags(t[:2**n])
    count = 2**n
    for i in range(1, 2**n):
        incr = 2*(2**n-i)
        matrix += sy.sparse.diags(t[count:count+incr:2], -i)+1j*sy.sparse.diags(t[count+1:count+incr+1:2], -i)
        count += incr
    return matrix.conjugate().T@matrix