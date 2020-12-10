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
def sum_qubits(a_bit, b_bit, carry_in):
    q0, q1, q2, q3, q4, q5, q6, q7 = zero, zero, zero, zero, zero, zero, zero, zero

    q0 = assign_bit(q0, a_bit)
    q1 = assign_bit(q1, b_bit)
    q2 = assign_bit(q2, carry_in)

    # AND1
    q3 = apply_toffoli(q0, q1, q3)
    # XOR1
    q4 = apply_cnot(q0, q4)
    q4 = apply_cnot(q1, q4)
    # XOR2
    q5 = apply_cnot(q2, q5)
    q5 = apply_cnot(q4, q5) # sum
    # AND2
    q6 = apply_toffoli(q2, q4, q6)
    # OR
    q3 = np.dot(pauli_x, q3)
    q6 = np.dot(pauli_x, q6)
    q7 = apply_toffoli(q3, q6, q7)
    q7 = np.dot(pauli_x, q7) # carry out

    sum_qubit   = measure([a[0] for a in q5])
    carry_qubit = measure([a[0] for a in q7])
    sum_bit,    = sum_qubit
    carry_bit,  = carry_qubit

    return sum_bit, carry_bit


if __name__=="__main__":
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


    print("Бүхэл тоо оруулна уу!")
    a = int(input("a="))
    b = int(input("b="))
    n_bits = 8
    a_bits = '{:08b}'.format(a)[-n_bits:]
    b_bits = '{:08b}'.format(b)[-n_bits:]

    print("########### ХОЁР БҮХЭЛ ТООНЫ НИЙЛБЭР ##############")
    result = ''
    carry_in = 0
    for i in reversed(range(n_bits)):
        a_bit    = int(a_bits[i])
        b_bit    = int(b_bits[i])
        summed, carry_out = sum_qubits(a_bit=a_bit, b_bit=b_bit, carry_in=carry_in)
        carry_in = carry_out
        result  += str(summed)
    result     = result[::-1]
    result_int = int(result, 2)

    print("{}={} \n{}={}".format(a, a_bits, b, b_bits))
    print("{}+{}={}".format(a_bits, b_bits, result))
    print("{}={}".format(result, result_int))
    print("########### ҮР ДҮН #################################")
    print("{}+{}={}".format(a, b, result_int))
    pass
