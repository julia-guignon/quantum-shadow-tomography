import numpy as np
import qiskit.quantum_info as qi
import qiskit.circuit as qc
import qiskit_aer as aer
from tqdm import tqdm

import scipy as sy

from src.qu_utils import *
from src.circuit_logic import *


def median_of_means_fidelity(cl_shadow, K):
    N = len(cl_shadow)
    fidelities = np.zeros(K)
    for k in range(1, K+1):
        fidelities[k-1] = np.mean(cl_shadow[(k-1)*N//K:k*N//K+1])
    # print(f'fidelities: {fidelities}')
    return np.median(fidelities)


def classical_shadow_clifford(n, N, noise=False, nshots=1, simulator=None, verbose=True):
    outcomes = []
    if not simulator:
        if noise:
            simulator = get_noisy_QASM()
        else:
            simulator = aer.QasmSimulator()

    for i in tqdm(range(N), disable=(not verbose)):
        # print('cc', n)
        clifford_op = qi.random_clifford(int(n))
        if nshots>1:
            counts = single_measurement_clifford(n, clifford_op, nshots=nshots, simulator=simulator)
            # keep only the highest val.
            counts = {max(counts, key=counts.get): counts[max(counts, key= counts.get)]}
        else:
            counts = single_measurement_clifford(n, clifford_op, nshots=1, simulator=simulator)
        bitstring = list(counts.keys())[0]
        # print(counts)
        outcomes.append({'clifford': clifford_op, 'bitstring': bitstring})

    return measurement_to_fidelity(n, outcomes)


def measurement_to_fidelity(n, outcomes):
    GHZ = init_GHZ(n)
    amplitudes = np.zeros(len(outcomes))
    for i, outcome in enumerate(outcomes):
        rotated_GHZ = GHZ.compose(outcome['clifford'].to_circuit())
        stabilizer_circuit = qi.StabilizerState(rotated_GHZ)
        amplitude = stabilizer_circuit.probabilities_dict_from_bitstring(outcome_bitstring=outcome['bitstring'])
        amplitudes[i] = list(amplitude.values())[0]
    return (2**n+1)*amplitudes-1


