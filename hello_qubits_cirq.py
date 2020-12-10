import cirq

q0 = cirq.GridQubit(0, 0)
q1 = cirq.GridQubit(0, 1)

circuit = cirq.Circuit(
        cirq.X(q0)**0.5,
        cirq.X(q1)**0.5,
        cirq.measure(q0, key='m0'),
        cirq.measure(q1, key='m1')
        )

print(circuit)


simulator = cirq.Simulator()
result = simulator.run(circuit, repetitions=20)

print("Үр дүн:")
print(result)
