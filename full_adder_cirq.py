import cirq

q0, q1, q2, q3, q4, q5, q6, q7 = cirq.LineQubit.range(8)

circuit = cirq.Circuit()

# Оролтын битүүд
circuit.append(cirq.X(q0)) # A=1
#circuit.append(cirq.X(q1)) # B=1
#circuit.append(cirq.X(q2)) # Cin=1

# AND1
circuit.append(cirq.TOFFOLI(q0, q1, q3))
# XOR1
circuit.append(cirq.CNOT(q0, q4))
circuit.append(cirq.CNOT(q1, q4))
# XOR2
circuit.append(cirq.CNOT(q2, q5))
circuit.append(cirq.CNOT(q4, q5))
# AND2
circuit.append(cirq.TOFFOLI(q2, q4, q6))
# OR
circuit.append(cirq.X(q3))
circuit.append(cirq.X(q6))
circuit.append(cirq.TOFFOLI(q3, q6, q7))
circuit.append(cirq.X(q7))

# Үр дүнг хэмжих
# SUM=q5, Cout=q7
circuit.append(cirq.measure(q5, q7, key='results'))

print(circuit)

sim = cirq.Simulator()
result = sim.run(circuit, repetitions=10000)

for sample in result.measurements['results'][:10]:
    print(sample)
