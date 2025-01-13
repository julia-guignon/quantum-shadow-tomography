import qiskit.quantum_info as qi
import qiskit.circuit as qc
import scipy as sy
import numpy as np
import itertools as it

def get_expectation_value(counts, basis):
    '''
        Returns the exepectation value for the given basis.
    '''
    expectation_value = 0
    nshots = sum(list(counts.values()))
    # print(f'basis: {basis}')
    for key, val in counts.items():
        parity = 1
        # print(f'key={key}, val={val}')
        for qubit, sigma in zip(key[::-1], basis):
            # print(sigma, int(qubit))
            if sigma!='I' and int(qubit):
                parity *= -1
            else:
                parity *= 1
            # print(parity)
        expectation_value += parity*val
    return expectation_value/nshots


to_Pauli = {'X': qi.Pauli('X').to_matrix(sparse=True),
            'Y': qi.Pauli('Y').to_matrix(sparse=True),
            'Z': qi.Pauli('Z').to_matrix(sparse=True),
            'I': qi.Pauli('I').to_matrix(sparse=True)}

            
def get_pauli_str_mat(string, to_Pauli=to_Pauli): # works
    '''
        Returns the sparse matrix representing the Pauli string.

        Input:
            - string (list of str)
            - to_Pauli (dict): given a str among {'X', 'Y', 'Z', 'I'}, provides the corresponding sparse matrix in csr format.

        Output:
            - matrix in csr format
    '''
    mat = to_Pauli[string[0]]
    for s in string[1:]:
        mat = sy.sparse.kron(mat, to_Pauli[s])
    return mat


def get_expectation_value_from_str(state, pauli_str):
    return (get_pauli_str_mat(pauli_str)@state).trace()


def init_GHZ(n): # works
    '''
        Prepare the GHZ state for n qubits.
    ''' 
    if n<1:
        raise Exception(f'The number of qubits must be greater or equal to 1 (you entered n={n}).')

    circuit = qc.QuantumCircuit(n, n)
    circuit.h(0)
    for i in range(n-1):
        circuit.cx(i, i+1)

    return circuit

def theoritical_GHZ(n):
    ghz = np.zeros((2**n, 1))
    ghz[0] = 1
    ghz[-1] = 1
    return ghz@ghz.T/2

    # ghz = sy.sparse.csr_matrix((2**n, 1))  
    # ghz[0, 0] = 1/np.sqrt(2)  
    # ghz[-1, 0] = 1/np.sqrt(2) 
    
    # if sparse:
    #     return ghz@ghz.T
    # else:
    #     return (ghz@ghz.T).todense()


def pauli_str_permutations(n, measurement_basis=['I', 'X', 'Y', 'Z']):
    '''
        Return a list of tuple of str containing all the possible permutations of Pauli strings with a length of n. 
        Credits to Emilio Peláez for this one (https://github.com/epelaaez/QuantumLibrary/blob/master/challenges/QOSF%20Monthly/July%202021.ipynb).
        
        //!\\ PRO TIP: the output scales as 4**n. If you have a bit of respect for your computer or you are unpatient, 
                       do not take n bigger than 9 (speaking with experience).
    '''

    possibilities = []
    combinations = it.combinations_with_replacement(measurement_basis, n)
    for comb in combinations:
        for item in set(list(it.permutations(comb))):
            possibilities.append(item)
    possibilities.remove(('I',)*n)
    assert len(possibilities)==4**n-1
    return possibilities

def pauli_str_permutations_old(n):
    '''
        Return a list of list of str containing all the possible permutations of Pauli strings with a length of n. 
        My first implementation of the function. Not optimized. Crashes for n>=10.

        //!\\ PRO TIP: the output scales as 3**n. If you don't wanna kill your computer, 
                     don't take n bigger than 9 (speaking with experience).
    '''
    possibilities = []
    for i in range(n+1):
        for j in range(n+1-i):
            k = n-i-j
            possibilities.append([p for p in it.permutations(['X']*i+['Y']*j+['Z']*k)])

    possibilities = list(set(sum(possibilities, [])))
    
    # assert len(possibilities)==3**n
    return possibilities


def make_it_semi_positive_definite(mat):
    eigvals, v = sy.linalg.eigh(mat)
    if np.all(eigvals>=0):
        return mat
    else:
        eigvals[eigvals<0] = 0
        eigvals = np.real(eigvals)
        result = v@np.diag(eigvals)@v.conj().T
        return result/np.trace(result)
        