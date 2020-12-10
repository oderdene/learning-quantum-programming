import numpy as np
import sympy as sp
from sympy.physics.quantum.qubit import matrix_to_qubit, Qubit
from sympy.physics.quantum.represent import represent


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


def AND(first_bit, second_bit):
    q0 = zero
    q1 = zero
    if first_bit==1:
        q0 = np.dot(pauli_x, q0)
    if second_bit==1:
        q1 = np.dot(pauli_x, q1)

    q_target   = zero
    q_combined = n_kron(q0, q1, q_target)

    result = np.dot(toffoli, q_combined)
    _, _, target_bit= list(matrix_to_qubit(result).free_symbols)[0].qubit_values

    return target_bit


if __name__=="__main__":
    res = AND(0, 0)
    print("AND(|00>)=", res)
    res = AND(0, 1)
    print("AND(|01>)=", res)
    res = AND(1, 0)
    print("AND(|10>)=", res)
    res = AND(1, 1)
    print("AND(|11>)=", res)

