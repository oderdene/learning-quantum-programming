from sympy.physics.quantum.qubit     import Qubit
from sympy.physics.quantum.qubit     import matrix_to_qubit
from sympy.physics.quantum.qubit     import IntQubit
from sympy.physics.quantum.qubit     import QubitBra
from sympy.physics.quantum.dagger    import Dagger
from sympy.physics.quantum.represent import represent
import numpy as np


def normalize_state(state):
    return state/np.linalg.norm(state)

def n_kron(*args):
    result = np.array([[1.+0.j]])
    for op in args:
        result = np.kron(result, op)
    return result


zero = np.array([[1.+0.j],
                 [0.+0.j]])

one  = np.array([[0.+0.j],
                 [1.+0.j]])

print("> Sympy хэрэглэн qubit дүрслэх туршилтууд :\n")

print("qubit |5> буюу integer дүрслэл ")
q0 = IntQubit(5)
print("qubit int     :", q0)
print("qubit nqubits :", q0.nqubits)
print("qubit values  :", q0.qubit_values)

q1 = Qubit(q0)
print("sympy qubit дүрслэл :", q1, "\n")

print("IntQubit(1, 1)")
q2 = IntQubit(1, 1)
print(q2, Qubit(q2))
print("IntQubit(1,0,1,0,1)")
q3 = IntQubit(1,0,1,0,1)
print(q3, Qubit(q3))
print(q3.qubit_values)
print(Qubit(q3).qubit_values)

print("Dagger", q1, Dagger(q1))
ip = Dagger(q1)*q1
print(ip)
print(ip.doit())

print("Qubit-ээс numpy руу хөрвүүлэх")
q = Qubit('01')
print(q)
q_np = np.array(q)
print(q_np)
#q_np0 = np.array(represent(q), dtype=np.cfloat)
q_np0 = np.array(represent(q))
print(q_np0.shape)
print(q_np0)

print("Numpy-аас Qubit-рүү хөрвүүлэх")
# https://stackoverflow.com/questions/30018977/how-can-i-get-a-list-of-the-symbols-in-a-sympy-expression
new_q = matrix_to_qubit(q_np0)
print(new_q.free_symbols)
new_q = matrix_to_qubit(np.array([[1],[0],[1],[0]]))
print(new_q.free_symbols)
new_q = matrix_to_qubit(np.array([[1],[0],[0],[0]]))
print(new_q)
new_q = matrix_to_qubit(np.array([[0],[1],[0],[0]]))
print(new_q)
new_q = matrix_to_qubit(np.array([[0],[0],[1],[0]]))
print(new_q)
new_q = matrix_to_qubit(np.array([[0],[0],[0],[1]]))
print(new_q)
