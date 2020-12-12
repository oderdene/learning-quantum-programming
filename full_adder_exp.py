import random
import sys
import numpy as np
import sympy as sp
from sympy.physics.quantum.qubit import matrix_to_qubit

# N qubit системд quantum gate хэрэглэх талаархи лавлагаанууд
#
# - https://quantumcomputing.stackexchange.com/questions/13143/how-to-make-toffoli-gate-using-matrix-form-in-multi-qubits-system
# - https://quantumcomputing.stackexchange.com/questions/10098/how-to-represent-an-n-qubit-circuit-in-matrix-form/10106
# - https://quantumcomputing.stackexchange.com/questions/5179/how-to-construct-matrix-of-regular-and-flipped-2-qubit-cnot
# - https://cs.stackexchange.com/questions/48834/applying-a-multi-qubit-quantum-gate-to-specific-qubits
#
#


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

def apply_pauli_x(psi, loc, n=8):
    op_list      = [ID2]*n
    op_list[loc] = pauli_x
    op_matrix    = n_kron_list(op_list)
    return np.dot(op_matrix, psi)

# Хоёр qubit систем дээр
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


# Full Adder
#
#   Ерөнхийдөө XOR үйлдлээр хоёр битийг нэмэх бөгөөд сануулсан оронгийн
#   битийг мөн адил XOR үйлдлээр нэмнэ. AND үйлдлээр эхний нэмэгдхүүн
#   дээр болон сануулсан орон дээр нэмж сануулах орон үүсэх эсэхийг мэдэн
#   ялгаж авна. Гаралтын сануулах орон OR үйлдлээр нэгтгэгдэнэ.
#
#   Дэлгэрэнгүй диаграм :
#
#     - https://github.com/sharavsambuu/learning-quantum-programming/blob/master/images/fulladder.jpg
#
#
def sum_qubits(a_bit, b_bit, carry_in, n=8):
    # |ψ> = |00000000>
    psi = n_kron_list([zero]*n)

    if a_bit==1:
        psi = apply_pauli_x(psi, 0)
    if b_bit==1:
        psi = apply_pauli_x(psi, 1)
    if carry_in==1:
        psi = apply_pauli_x(psi, 2)

    # AND1
    psi = apply_toffoli(psi, 0, 1, 3)
    # XOR1
    psi = apply_cnot(psi, 0, 4)
    psi = apply_cnot(psi, 1, 4)
    # XOR2
    psi = apply_cnot(psi, 2, 5)
    psi = apply_cnot(psi, 4, 5) # sum
    # AND2
    psi = apply_toffoli(psi, 2, 4, 6)
    # OR
    psi = apply_pauli_x(psi, 3)
    psi = apply_pauli_x(psi, 6)
    psi = apply_toffoli(psi, 3, 6, 7)
    psi = apply_pauli_x(psi, 7) # carry out

    qubit         = matrix_to_qubit(psi)
    print(qubit)
    qubit_values  = list(qubit.free_symbols)[0].qubit_values
    _,_,_,_,_,sum_bit,_,carry_out = qubit_values

    return sum_bit, carry_out


if __name__=="__main__":

    # |11001>
    q11001 = n_kron(one, one, zero, zero, one)

    print("\n\nCNOT testing...\n")
    print("CNOT_0_2(|11001>) => |11101>")
    q0 = q11001
    q0 = apply_cnot(q0, 0, 2, 5)
    print(matrix_to_qubit(q0))
    print("CNOT_2_4(|11001>) => |11001>")
    q0 = q11001
    q0 = apply_cnot(q0, 2, 4, 5)
    print(matrix_to_qubit(q0))
    print("CNOT_1_4(|11001>) => |11000>")
    q0 = q11001
    q0 = apply_cnot(q0, 1, 4, 5)
    print(matrix_to_qubit(q0))
    print("CNOT_4_1(|11001>) => |10001>")
    q0 = q11001
    q0 = apply_cnot(q0, 4, 1, 5)
    print(matrix_to_qubit(q0))

    print("\n\nPauli-X testing...\n")
    print("X_0(|11001>) => |01001>")
    q0 = q11001
    q0 = apply_pauli_x(q0, 0, 5)
    print(matrix_to_qubit(q0))
    print("X_3(|11001>) => |11011>")
    q0 = q11001
    q0 = apply_pauli_x(q0, 3, 5)
    print(matrix_to_qubit(q0))
    print("X_4(|11001>) => |11000>")
    q0 = q11001
    q0 = apply_pauli_x(q0, 4, 5)
    print(matrix_to_qubit(q0))

    print("\n\nToffoli testing...\n")
    print("Toffoli_0_1_2(|11001>) => |11101>")
    q0 = q11001
    q0 = apply_toffoli(q0, 0, 1, 2, 5)
    print(matrix_to_qubit(q0))
    print("Toffoli_1_4_2(|11001>) => |11101>")
    q0 = q11001
    q0 = apply_toffoli(q0, 1, 4, 2, 5)
    print(matrix_to_qubit(q0))
    print("Toffoli_0_2_3(|11001>) => |11001>")
    q0 = q11001
    q0 = apply_toffoli(q0, 0, 2, 3, 5)
    print(matrix_to_qubit(q0))


    sys.exit(0)
    print("Full adder бит нэмэх хүснэгт:")

    a_bit, b_bit, carry_in = 1, 1, 1
    summed, carry_out = sum_qubits(a_bit=a_bit, b_bit=b_bit, carry_in=carry_in)
    print("a={} b={} carry_in={} sum={} carry_out={}".format(a_bit, b_bit, carry_in, summed, carry_out))

    a_bit, b_bit, carry_in = 0, 1, 1
    summed, carry_out = sum_qubits(a_bit=a_bit, b_bit=b_bit, carry_in=carry_in)
    print("a={} b={} carry_in={} sum={} carry_out={}".format(a_bit, b_bit, carry_in, summed, carry_out))

    a_bit, b_bit, carry_in = 1, 0, 1
    summed, carry_out = sum_qubits(a_bit=a_bit, b_bit=b_bit, carry_in=carry_in)
    print("a={} b={} carry_in={} sum={} carry_out={}".format(a_bit, b_bit, carry_in, summed, carry_out))

    a_bit, b_bit, carry_in = 0, 0, 1
    summed, carry_out = sum_qubits(a_bit=a_bit, b_bit=b_bit, carry_in=carry_in)
    print("a={} b={} carry_in={} sum={} carry_out={}".format(a_bit, b_bit, carry_in, summed, carry_out))

    a_bit, b_bit, carry_in = 1, 1, 0
    summed, carry_out = sum_qubits(a_bit=a_bit, b_bit=b_bit, carry_in=carry_in)
    print("a={} b={} carry_in={} sum={} carry_out={}".format(a_bit, b_bit, carry_in, summed, carry_out))

    a_bit, b_bit, carry_in = 0, 1, 0
    summed, carry_out = sum_qubits(a_bit=a_bit, b_bit=b_bit, carry_in=carry_in)
    print("a={} b={} carry_in={} sum={} carry_out={}".format(a_bit, b_bit, carry_in, summed, carry_out))

    a_bit, b_bit, carry_in = 1, 0, 0
    summed, carry_out = sum_qubits(a_bit=a_bit, b_bit=b_bit, carry_in=carry_in)
    print("a={} b={} carry_in={} sum={} carry_out={}".format(a_bit, b_bit, carry_in, summed, carry_out))

    a_bit, b_bit, carry_in = 0, 0, 0
    summed, carry_out = sum_qubits(a_bit=a_bit, b_bit=b_bit, carry_in=carry_in)
    print("a={} b={} carry_in={} sum={} carry_out={}".format(a_bit, b_bit, carry_in, summed, carry_out))

    pass

