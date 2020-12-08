import numpy as np


def normalize_state(state):
    return state/np.linalg.norm(state)

def n_kron(*args):
    result = np.array([[1.+0.j]])
    for q in args:
        result = np.kron(result, q)
    return result


# |0>
zero = np.array([[1.+0.j],
                 [0.+0.j]])
print("|0>")
print(zero, "\n")

# |1>
one  = np.array([[0.+0.j],
                 [1.+0.j]])
print("|1>")
print(one, "\n")

# |+> = 1/√2(|0>+|1>)
plus = normalize_state(zero+one)
print("|+> = 1/√2(|0>+|1>)")
print(plus, "\n")

# |-> = 1/√2(|0>-|1>)
minus = normalize_state(zero-one)
print("|-> = 1/√2(|0>-|1>)")
print(minus, "\n")


# Quantum logic gates
# - https://en.wikipedia.org/wiki/Quantum_logic_gate

pauli_x = np.array(
        [[0.+0.j, 1.+0.j],
         [1.+0.j, 0.+0.j]]
        )

pauli_y = np.array(
        [[0.+0.j, -1j    ],
         [1j    ,  0.+0.j]]
        )

pauli_z = np.array(
        [[1.+0.j,  0.+0.j],
         [0.+0.j, -1.+0.j]]
        )

hadamard = np.array(
        [[1.+0.j,  1.+0.j],
         [1.+0.j, -1.+0.j]]
        )*1/np.sqrt(2)

ID = np.eye(2, dtype=np.cfloat)


# |0> болон |1> төлвүүдээр цэнэглэгдсэн qubit-үүд
q0 = zero
q1 = one

x_flipped_q0 = np.dot(pauli_x, q0)
print("Pauli-X(|0>)")
print(x_flipped_q0, "\n")

x_flipped_q1 = np.dot(pauli_x, q1)
print("Pauli-X(|1>)")
print(x_flipped_q1, "\n")

y_flipped_q0 = np.dot(pauli_y, q0)
print("Pauli-Y(|0>)")
print(y_flipped_q0, "\n")

y_flipped_q1 = np.dot(pauli_y, q1)
print("Pauli-Y(|1>)")
print(y_flipped_q1, "\n")

z_flipped_q0 = np.dot(pauli_z, q0)
print("Pauli-Z(|0>)")
print(z_flipped_q0, "\n")

z_flipped_q1 = np.dot(pauli_z, q1)
print("Pauli-Z(|1>)")
print(z_flipped_q1, "\n")

hadamard_q0 = np.dot(hadamard, q0)
print("Hadamard(|0>) = |+> = 1/√2(|0>+|1>)")
print(hadamard_q0, "\n")

hadamard_q1 = np.dot(hadamard, q1)
print("Hadamard(|1>) = |-> = 1/√2(|0>-|1>)")
print(hadamard_q1, "\n")


# |001> = |0>⊗ |0>⊗ |1>
print("##### Олон qubit-үүдийн төлөв \n")

# |00>
q00 = np.kron(zero, zero)
print("|00>")
print(q00, "\n")

# |01>
q01 = np.kron(zero, one)
print("|01>")
print(q01, "\n")

# |10>
q10 = np.kron(one, zero)
print("|10>")
print(q10, "\n")

# |11>
q11 = np.kron(one, one)
print("|11>")
print(q11, "\n")

# |-->
q_minusminus = np.kron(minus, minus)
print("|-->")
print(q_minusminus, "\n")

# |-+>
q_minusplus = np.kron(minus, plus)
print("|-+>")
print(q_minusplus, "\n")

# |+->
q_plusminus = np.kron(plus, minus)
print("|+->")
print(q_plusminus, "\n")

# |++>
q_plusplus = np.kron(plus, plus)
print("|++>")
print(q_plusplus, "\n")

# 1/√2(|00>+|11>)
q_catstate = normalize_state(q00+q11)
print("1/√2(|00>+|11> cat state")
print(q_catstate, "\n")

# |000>
q000 = n_kron(zero, zero, zero)
print("|000>")
print(q000, "\n")

# |001>
q001 = n_kron(zero, zero, one)
print("|001>")
print(q001, "\n")

# |010>
q010 = n_kron(zero, one, zero)
print("|010>")
print(q010, "\n")

# |111>
q111 = n_kron(one, one, one)
print("|111>")
print(q111, "\n")

# H0(|101>) Эхний qubit дээр Hadamard хэрэглэх
H0_on_3 = n_kron(hadamard, ID, ID)
q101    = n_kron(one, zero, one)
H0_101  = np.dot(H0_on_3, q101)
print("H0(|101>)")
print(H0_101, "\n")

# H1(|100>) Хоёр дахь qubit дээр Hadamard хэрэглэх
H1_on_3 = n_kron(ID, hadamard, ID)
q100    = n_kron(one, zero, zero)
H1_100  = np.dot(H1_on_3, q100)
print("H1(|100>)")
print(H1_100, "\n")

# H2(|001>) Гурав дахь qubit дээр Hadamard хэрэглэх
H2_on_3 = n_kron(ID, ID, hadamard)
q001    = n_kron(one, one, zero)
H2_001  = np.dot(H2_on_3, q001)
print("H2(|001>)")
print(H2_001, "\n")



# CNOT gate буюу квант IF хэрэгжүүлэх
# P0 = |0><0|, P1 = |1><1|
# CNOT = P0⊗ ID + P1⊗ X
#
#  0: ───@───
#        │
#  1: ───X───
#
P0        = np.dot(zero, zero.T)
P1        = np.dot(one , one.T )
CNOT_on_2 = n_kron(P0, ID) + n_kron(P1, pauli_x)

# CNOT(|00>)
CNOT_00 = np.dot(CNOT_on_2, q00)
print("CNOT(|00>)")
print(CNOT_00, "\n")

# CNOT(|01>)
CNOT_01 = np.dot(CNOT_on_2, q01)
print("CNOT(|01>)")
print(CNOT_01, "\n")

# CNOT(|10>)
CNOT_10 = np.dot(CNOT_on_2, q10)
print("CNOT(|10>)")
print(CNOT_10, "\n")

# CNOT(|11>)
CNOT_11 = np.dot(CNOT_on_2, q11)
print("CNOT(|11>)")
print(CNOT_11, "\n")


# Bell state буюу квант орооцолдоон(aka quantum entanglement)
#
#  |0> ───H───@───
#         │   │
#  |0> ───X───M───
#
# 1/√2(|00>+|11>)
#
H0_on_2    = n_kron(hadamard, ID) # 2 qubit-н эхнийх дээр Hadamard
P0         = np.dot(zero, zero.T)
P1         = np.dot(one , one.T )
CNOT_on_2  = n_kron(P0, ID) + n_kron(P1, pauli_x)

q0         = zero
q1         = zero
q00        = np.kron(q0, q1)
H0_00      = np.dot(H0_on_2, q00)
CNOT_H0_00 = np.dot(CNOT_on_2, H0_00)
print("BellState(|0>, |0>) = 1/√2(|00>+|11>")
print(CNOT_H0_00, "\n")
