import cirq

# Generate Bell State for 4 state


q0, q1 = cirq.LineQubit.range(2)

circuit = cirq.Circuit(
        cirq.H(q0),
        cirq.H(q1),
        cirq.CNOT(q0, q1),
        cirq.measure(q0, q1, key='result'),
        )

print(circuit)


simulator = cirq.Simulator()
print("квант код ажиллуулж байна түр хүлээгээрэй...")
result = simulator.run(circuit, repetitions=100000)

for sample in result.measurements['result'][:10]:
    print(sample)

print(result.histogram(key='result'))
