import sys
import cirq

print("Бүхэл тоо оруулна уу!")

a = int(input("a="))
b = int(input("b="))

print("a={} b={}".format(a, b))

n_bits = 3
a_bits = '{:08b}'.format(a)[-n_bits:]
b_bits = '{:08b}'.format(b)[-n_bits:]

print("a битүүд = {}".format(a_bits))
print("b битүүд = {}".format(b_bits))


qubits   = cirq.LineQubit.range(6)
a_qubits = qubits[0:3]
b_qubits = qubits[3:6]

def assign_bits(bits, *qubits):
    #for bit, qubit in zip(bits, list(qubits)[::-1]):
    for bit, qubit in zip(bits, list(qubits)):
        if bit=='1':
            yield cirq.X(qubit) # Pauli-X, Not gate

circuit = cirq.Circuit(
        assign_bits(a_bits, *a_qubits),
        assign_bits(b_bits, *b_qubits),
        cirq.measure(*a_qubits, key='a_qubits'),
        cirq.measure(*b_qubits, key='b_qubits')
        )
print(circuit)


simulator    = cirq.Simulator()
measurements = simulator.run(circuit, repetitions=20).measurements
print("a_qubits :")
print(measurements['a_qubits'][:5])
print("b_qubits :")
print(measurements['b_qubits'][:5])

