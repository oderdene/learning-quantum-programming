import time
import cirq

q0, q1, q2 = cirq.LineQubit.range(3)

circuit = cirq.Circuit()

# Оролтын битүүд
circuit.append(cirq.X(q0))
circuit.append(cirq.X(q1))

# OR gate
circuit.append(cirq.X(q0))
circuit.append(cirq.X(q1))
circuit.append(cirq.TOFFOLI(q0, q1, q2))
circuit.append(cirq.X(q2))

circuit.append(cirq.measure(q2, key='qubits'))

print(circuit)

sim = cirq.Simulator()
result = sim.run(circuit, repetitions=10000)

for sample in result.measurements['qubits'][:10]:
    print("1 1 ",sample)

