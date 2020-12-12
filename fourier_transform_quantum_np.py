import random
import sys
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


# Qubit measurement
#
#   Qubit-ийн amplitude-ийн дагуу классик бит гаргаж авах
#   тухайн битрүү collapse хийгдэж ирэх магадлал amplitude-аар илэрхийлэгдэнэ
#
#
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

# N qubit системд хэрэглэх NOT
#
#   apply_pauli_x(|10001>, 2, n=5) => |10101>
#   apply_pauli_x(|10101>, 4, n=5) => |10100>
#
#
def apply_pauli_x(psi, loc, n=8):
    op_list      = [ID2]*n
    op_list[loc] = pauli_x
    op_matrix    = n_kron_list(op_list)
    return np.dot(op_matrix, psi)

# Хоёр qubit систем дээр
#
#   P0 = |0><0|, P1 = |1><1|
#   CNOT = P0⊗ ID + P1⊗ X
#
# Олон qubit систем дээр жишээ нь 10 qubit систем дээр 3-р qubit нь
# control qubit, 9-р qubit нь target qubit бол CNOT матриц дараах
# байдлаар үүсгэгдэнэ.
#
#   Хэрэв 3-р qubit |0> бол 9-р qubit-ийг хэвээр үлдээнэ
#     I⊗I⊗|0⟩⟨0|⊗I⊗I⊗I⊗I⊗I⊗I⊗I
#   эсрэгээрээ |1> бол
#     I⊗I⊗|1⟩⟨1|⊗I⊗I⊗I⊗I⊗I⊗X⊗I
#
#
def apply_cnot(psi, control_loc, target_loc, n=8):
    P0                     = np.dot(zero, zero.T)
    P1                     = np.dot(one , one.T )
    op_list_0              = [ID2]*n
    op_list_0[control_loc] = P0
    op_list_1              = [ID2]*n
    op_list_1[control_loc] = P1
    op_list_1[ target_loc] = pauli_x
    op_matrix = n_kron_list(op_list_0)+n_kron_list(op_list_1)
    return np.dot(op_matrix, psi)

# N qubit системд хэрэглэх SWAP
#
#   apply_swap(|10110>, 1, 3, n=5) => |11100>
#   apply_swap(|11010>, 0, 4, n=5) => |01011>
#
#
def apply_swap(psi, a_loc, b_loc, n=8):
    psi = apply_cnot(psi, a_loc, b_loc, n=n)
    psi = apply_cnot(psi, b_loc, a_loc, n=n)
    psi = apply_cnot(psi, a_loc, b_loc, n=n)
    return psi


# Quantum fourier transform
def fourier_transform():
    psi = n_kron_list([zero]*4)
    print("\n\nQuantum fourier transform\n")
    print(psi)
    pass


if __name__=="__main__":
    fourier_transform()
