import numpy as np


zero = np.array([[1.+0.j],
                 [0.+0.j]]) # |0>
one  = np.array([[0.+0.j],
                 [1.+0.j]]) # |1>


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


# |0> болон |1> төлвүүдээр цэнэглэгдсэн qubit-үүд
q0 = zero
q1 = one

print("q0=|0>")
print(q0, "\n")

print("q1=|1>")
print(q1, "\n")

x_flipped_q0 = np.dot(pauli_x, q0)
print("x flipped q0")
print(x_flipped_q0, "\n")

x_flipped_q1 = np.dot(pauli_x, q1)
print("x flipped q1")
print(x_flipped_q1, "\n")

y_flipped_q0 = np.dot(pauli_y, q0)
print("y flipped q0")
print(y_flipped_q0, "\n")

y_flipped_q1 = np.dot(pauli_y, q1)
print("y flipped q1")
print(y_flipped_q1, "\n")

z_flipped_q0 = np.dot(pauli_z, q0)
print("z flipped q0")
print(z_flipped_q0, "\n")

z_flipped_q1 = np.dot(pauli_z, q1)
print("z flipped q1")
print(z_flipped_q1, "\n")

hadamard_q0 = np.dot(hadamard, q0)
print("hadamard q0")
print(hadamard_q0, "\n")

hadamard_q1 = np.dot(hadamard, q1)
print("hadamard q1")
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
