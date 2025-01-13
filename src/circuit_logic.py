import numpy as np
import qiskit.quantum_info as qi
import qiskit.circuit as qc
import qiskit_aer as aer
import qiskit_aer.noise as noise
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.providers.fake_provider import GenericBackendV2
from tqdm import tqdm

import scipy as sy

from src.qu_utils import *
from src.py_utils import *
from src.ML_utils import *


def measurement_in_some_basis(circuit, basis): # works
    '''
        circuit (qiskit.circuit.QuantumCircuit)
        basis (tuple of str): tuple of str representing the basis of measurement of each qubit. 
                             Accepted values for the str: ('I', 'X', 'Y' or 'Z').
    '''
    n_qu = circuit.num_qubits
    n_cl = circuit.num_clbits

    if n_qu!=n_cl:
        raise Exception(f'The classical register (num_clbits={n_cl}) must have the same size as the quantum register (num_qubits={n_qu}).')

    if len(basis)!=n_qu:
        raise Exception(f'You must provide the same number of basis you want to measure in as the number of qubits of the circuit. You provided {len(basis)} basis for {n_qu} qubits.')
    
    if basis.count('I')+basis.count('X')+basis.count('Y')+basis.count('Z')!=n_qu:
        raise Exception('The only accepted measurement basis are I, X, Y or Z.')

    for i in range(n_qu):
        if basis[i]=='X':
            # to measure in X basis
            circuit.h(i)
        elif basis[i]=='Y':
            # to measure in Y basis
            circuit.sdg(i)
            circuit.h(i)
            
    circuit.measure_all(add_bits=False)



def get_noisy_QASM(backend='ibm_kyiv'):
    service = QiskitRuntimeService(
        channel='ibm_quantum',
        instance='ibm-q/open/main',
    )

    # fake_backend = GenericBackendV2(num_qubits=3)
    # noise_model = noise.NoiseModel.from_backend(fake_backend)
    
    backend = service.backend(backend)
    noise_model = noise.NoiseModel.from_backend(backend)

    noisy_qasm_simulator  = aer.QasmSimulator(noise_model=noise_model)
    return noisy_qasm_simulator


def run_circuit(simulator, circuit, shots=1000, method=None):
    if method:
        results = simulator.run(circuit, shots=shots, method=method).result()
    else:
        results = simulator.run(circuit, shots=shots).result()
    return results
    

def measurement_process(n, nshots, noise, simulator, verbose):
    pauli_strs = pauli_str_permutations(n)
    assert len(pauli_strs)==4**n-1

    if simulator==None:
        if noise:
            simulator = get_noisy_QASM()
        else:
            simulator = aer.QasmSimulator()

    done = []
    results = []
    measurements = {}
    expectation_values = {}
    for i in tqdm(range(len(pauli_strs)), disable=(not verbose)):
        tmp_str = replace_in_list(pauli_strs[i], 'I', 'Z')
        if tmp_str in done:
            expectation_values[pauli_strs[i]] = get_expectation_value(results[done.index(tmp_str)], pauli_strs[i])
            measurements[pauli_strs[i]] = results[done.index(tmp_str)]
        else:  
            tmp_results = single_measurement(n, tmp_str, nshots, simulator)
            expectation_values[pauli_strs[i]] = get_expectation_value(tmp_results, pauli_strs[i])
            done.append(tmp_str)
            results.append(tmp_results)
            measurements[pauli_strs[i]] = tmp_results
    
    return expectation_values, measurements

def single_measurement(n, pauli_str, nshots, simulator):
    circuit = init_GHZ(n)
    measurement_in_some_basis(circuit, pauli_str)
    counts = run_circuit(simulator, circuit, nshots).get_counts()
    return counts

   