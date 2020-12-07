import cirq

# 4 qubit-үүдээр Bellstate болон Hadamard ашиглан сэлгэмэл үүсгэх
# |0000> |0001> |0010> ... |1110> |1111>
# нийт 2^4=16-н төлөв үүснэ

q0, q1, q2, q3 = cirq.LineQubit.range(4)

circuit = cirq.Circuit(
        cirq.H(q0),
        cirq.H(q1),
        cirq.H(q2),
        cirq.H(q3),
        cirq.CNOT(q0, q1),
        cirq.CNOT(q1, q2),
        cirq.CNOT(q2, q3),
        cirq.measure(q0, q1, q2, q3, key='result'),
        )

print(circuit)


simulator = cirq.Simulator()
print("квант код ажиллуулж байна түр хүлээгээрэй...")
result = simulator.run(circuit, repetitions=100000)

for sample in result.measurements['result'][:10]:
    print(sample)

print(result.histogram(key='result'))
