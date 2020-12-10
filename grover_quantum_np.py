import random
import numpy as np
import sympy as sp
from sympy.physics.quantum.qubit import matrix_to_qubit


def normalize_state(state):
    return state/np.linalg.norm(state)

def n_kron(*args):
    result = np.array([[1.+0.j]])
    for op in args:
        result = np.kron(result, op)
    return result

# |0>
zero = np.array([[1.+0.j],
                 [0.+0.j]], dtype=np.cfloat)
# |1>
one  = np.array([[0.+0.j],
                 [1.+0.j]], dtype=np.cfloat)
# |-> = 1/√2(|0>-|1>)
minus = normalize_state(zero-one)
# |+> = 1/√2(|0>+|1>)
plus  = normalize_state(zero+one)

# Quantum logic gates
# - https://en.wikipedia.org/wiki/Quantum_logic_gate
pauli_x = np.array(
        [[0.+0.j, 1.+0.j],
         [1.+0.j, 0.+0.j]])
pauli_y = np.array(
        [[0.+0.j, -1j    ],
         [1j    ,  0.+0.j]])
pauli_z = np.array(
        [[1.+0.j,  0.+0.j],
         [0.+0.j, -1.+0.j]])
hadamard = np.array(
        [[1.+0.j,  1.+0.j],
         [1.+0.j, -1.+0.j]]
        )*1/np.sqrt(2.+0.j)
ID2 = np.eye(2, dtype=np.cfloat)
toffoli = np.array(
        [[1, 0, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 1, 0]],
        dtype=np.cfloat)


def measure(amplitudes, repetitions=10):
    measurements = []
    for _ in range(repetitions):
        weights   = [abs(amplitude)**2 for amplitude in amplitudes]
        outcome   = random.choices(range(len(amplitudes)), weights)[0]
        new_state = np.zeros((len(amplitudes), 1))
        new_state[outcome][0] = 1
        measurements.append(new_state)
    sample = random.choice(measurements)
    qubit  = list(matrix_to_qubit(np.array(sample)).free_symbols)[0]
    return qubit.qubit_values

def assign_bit(qubit, bit):
    if bit==1:
        return np.dot(pauli_x, qubit)
    return qubit

def apply_toffoli(q0, q1, q_target):
    q_combined = n_kron(q0, q1, q_target)
    new_state  = np.dot(toffoli, q_combined)
    qubit_values = list(matrix_to_qubit(new_state).free_symbols)[0].qubit_values
    _, _, updated_target = qubit_values
    q_target = (lambda x: zero if x==0 else one)(updated_target)
    return q_target

def apply_cnot(q0, q_target):
    P0         = np.dot(zero, zero.T)
    P1         = np.dot(one , one.T )
    CNOT_on_2  = n_kron(P0, ID2) + n_kron(P1, pauli_x)
    q_combined    = n_kron(q0, q_target)
    CNOT_0_target = np.dot(CNOT_on_2, q_combined)
    qubit_values  = list(matrix_to_qubit(CNOT_0_target).free_symbols)[0].qubit_values
    _, updated_target = qubit_values
    q_target = (lambda x: zero if x==0 else one)(updated_target)
    return q_target


# Grover-ийн алгоритм
#
# Лавлагаа :
#  - https://en.wikipedia.org/wiki/Grover%27s_algorithm
#
#

def grover():
    print("|00> төлөв хайхад зориулагдсан oracle")
    q0, q1 = zero, zero
    q0 = np.dot(hadamard, q0)
    q1 = np.dot(hadamard, q1)

    q0_measure, = measure([a[0] for a in q0])
    q1_measure, = measure([a[0] for a in q1])
    print("measure |q0q1> = |{}{}>".format(q0_measure, q1_measure))
    pass




if __name__=="__main__":
    print("Grover's algorithm")
    grover()
    pass
