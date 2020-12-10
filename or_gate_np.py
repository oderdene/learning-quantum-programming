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
    q_bit  = list(matrix_to_qubit(np.array(sample)).free_symbols)[0]

    return q_bit.qubit_values


# OR circuit загвар
#
# q0       : ──X──@─────────
#                 |
# q1       : ──X──@─────────
#                 |
# q_target : ─────X──X──M───
#
#

def QOR(first_bit, second_bit):
    q0 = zero
    q1 = zero
    if first_bit==1:
        q0 = np.dot(pauli_x, q0)
    if second_bit==1:
        q1 = np.dot(pauli_x, q1)
    q_target   = zero

    # OR circuit
    q0         = np.dot(pauli_x, q0)
    q1         = np.dot(pauli_x, q1)
    q_combined = n_kron(q0, q1, q_target)

    new_state  = np.dot(toffoli, q_combined)

    _, _, q3 = list(matrix_to_qubit(new_state).free_symbols)[0].qubit_values
    q_target = (lambda x: zero if x==0 else one)(q3)

    q_target = np.dot(pauli_x, q_target)

    qubits  = measure([a[0] for a in q_target])
    result, = qubits
    return result


if __name__=="__main__":
    res = QOR(0, 0)
    print("OR(|00>)=", res)
    res = QOR(0, 1)
    print("OR(|01>)=", res)
    res = QOR(1, 0)
    print("OR(|10>)=", res)
    res = QOR(1, 1)
    print("OR(|11>)=", res)

