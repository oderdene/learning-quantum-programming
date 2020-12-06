import time
import cirq

q0, q1, q2 = cirq.LineQubit.range(3)

circuit = cirq.Circuit()
circuit.append(cirq.X(q0)) # |0> -> |1>
circuit.append(cirq.X(q1)) # |0> -> |1>
circuit.append(cirq.TOFFOLI(q0, q1, q2)) # q2 := q0 AND q1
circuit.append(cirq.measure(q0, q1, q2, key='qubits'))

print(circuit)

sim = cirq.Simulator()
result = sim.run(circuit, repetitions=10000)

for sample in result.measurements['qubits'][:10]:
    print(sample)

