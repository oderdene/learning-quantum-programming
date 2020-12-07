import cirq

# 3 qubit-үүдээр Bellstate болон Hadamard ашиглан сэлгэмэл үүсгэх
# |000> |001> |010> |011> |100> |101> |110> |111>


q0, q1, q2 = cirq.LineQubit.range(3)

circuit = cirq.Circuit(
        cirq.H(q0),
        cirq.H(q1),
        cirq.H(q2),
        cirq.CNOT(q0, q1),
        cirq.CNOT(q1, q2),
        cirq.measure(q0, q1, q2, key='result'),
        )

print(circuit)


simulator = cirq.Simulator()
print("квант код ажиллуулж байна түр хүлээгээрэй...")
result = simulator.run(circuit, repetitions=100000)

for sample in result.measurements['result'][:10]:
    print(sample)

print(result.histogram(key='result'))
