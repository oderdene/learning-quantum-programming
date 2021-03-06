import random
import sys
from collections import Counter
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

def n_kron_list(args):
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
#
# - https://en.wikipedia.org/wiki/Quantum_logic_gate
#
#
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
swap = np.array(
        [[1, 0, 0, 0],
         [0, 0, 1, 0],
         [0, 1, 0, 0],
         [0, 0, 0, 1]],
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

def apply_pauli_x(psi, loc, n):
    op_list      = [ID2]*n
    op_list[loc] = pauli_x
    op_matrix    = n_kron_list(op_list)
    return np.dot(op_matrix, psi)

def apply_pauli_z(psi, loc, n):
    op_list      = [ID2]*n
    op_list[loc] = pauli_z
    op_matrix    = n_kron_list(op_list)
    return np.dot(op_matrix, psi)

def apply_hadamard(psi, loc, n):
    op_list      = [ID2]*n
    op_list[loc] = hadamard
    op_matrix    = n_kron_list(op_list)
    return np.dot(op_matrix, psi)

def apply_cnot(psi, control_loc, target_loc, n):
    P0                     = np.dot(zero, zero.T)
    P1                     = np.dot(one , one.T )
    op_list_0              = [ID2]*n
    op_list_0[control_loc] = P0
    op_list_1              = [ID2]*n
    op_list_1[control_loc] = P1
    op_list_1[ target_loc] = pauli_x
    op_matrix = n_kron_list(op_list_0)+n_kron_list(op_list_1)
    return np.dot(op_matrix, psi)

def apply_swap(psi, a_loc, b_loc, n):
    psi = apply_cnot(psi, a_loc, b_loc, n=n)
    psi = apply_cnot(psi, b_loc, a_loc, n=n)
    psi = apply_cnot(psi, a_loc, b_loc, n=n)
    return psi

def apply_cz(psi, control_loc, target_loc, n):
    P0                     = np.dot(zero, zero.T)
    P1                     = np.dot(one , one.T )
    op_list_0              = [ID2]*n
    op_list_0[control_loc] = P0
    op_list_1              = [ID2]*n
    op_list_1[control_loc] = P1
    op_list_1[ target_loc] = pauli_z
    op_matrix = n_kron_list(op_list_0)+n_kron_list(op_list_1)
    return np.dot(op_matrix, psi)

def apply_cz_rot(psi, control_loc, target_loc, rotation, n):
    P0                     = np.dot(zero, zero.T)
    P1                     = np.dot(one , one.T )
    op_list_0              = [ID2]*n
    op_list_0[control_loc] = P0
    op_list_1              = [ID2]*n
    op_list_1[control_loc] = P1
    op_list_1[ target_loc] = pauli_z**rotation
    op_matrix = n_kron_list(op_list_0)+n_kron_list(op_list_1)
    return np.dot(op_matrix, psi)

def apply_cz_and_swap(psi, a_loc, b_loc, rotation, n):
    psi = apply_cz_rot(psi, a_loc, b_loc, rotation, n=n)
    psi = apply_swap(psi, a_loc, b_loc, n=n)
    return psi


# Quantum fourier transform
#
#
#  ─H───@^0.5───×───H────────────@^0.5─────×───H──────────@^0.5──×─H
#       │       │                │         │               │     │
#  ─────@───────×───@^0.25───×───@─────────×───@^0.25───×──@─────×──
#                   │        │                 │        │
#  ─────────────────┼────────┼───@^0.125───×───┼────────┼───────────
#                   │        │   │         │   │        │
#  ─────────────────@────────×───@─────────×───@────────×───────────
#
#
# Лавлагаа:
#  - https://qiskit.org/textbook/ch-algorithms/quantum-fourier-transform.html
#
# Сүүлийн төлөв:
#
# [0.25+0.j 0.25+0.j 0.25+0.j 0.25+0.j 0.25+0.j 0.25+0.j 0.25+0.j 0.25+0.j
#  0.25+0.j 0.25+0.j 0.25+0.j 0.25+0.j 0.25+0.j 0.25+0.j 0.25+0.j 0.25+0.j]
#
#
def fourier_transform():
    psi = n_kron_list([zero]*4)

    print("\nQuantum fourier transform\n")

    psi = apply_hadamard(psi, 0, n=4)

    psi = apply_cz_and_swap(psi, 0, 1,   0.5, n=4)
    psi = apply_cz_and_swap(psi, 1, 2,  0.25, n=4)
    psi = apply_cz_and_swap(psi, 2, 3, 0.125, n=4)

    psi = apply_hadamard(psi, 0, n=4)

    psi = apply_cz_and_swap(psi, 0, 1,  0.5, n=4)
    psi = apply_cz_and_swap(psi, 1, 2, 0.25, n=4)

    psi = apply_hadamard(psi, 0, n=4)

    psi = apply_cz_and_swap(psi, 0, 1,  0.5, n=4)

    psi = apply_hadamard(psi, 0, n=4)


    measurements = []
    repetitions  = 10000
    for idx in range(0, repetitions):
        psi_values = measure([a[0] for a in psi], repetitions=1)
        psi_str    = "".join([str(bit) for bit in psi_values])
        measurements.append(psi_str)
        print(psi_str, "=>", int(psi_str, 2))
        pass

    print("\n", matrix_to_qubit(psi), "\n")

    histogram = Counter(measurements)
    print("\n", list(histogram.keys()), "\n")
    print("\n", histogram, "\n")

    pass


if __name__=="__main__":
    fourier_transform()
