import time
import cirq

q0, q1, q2, q3 = cirq.LineQubit.range(4)

circuit = cirq.Circuit()
circuit.append(cirq.H(q0))
circuit.append(cirq.H(q1))
circuit.append(cirq.H(q2))
circuit.append(cirq.H(q3))
circuit.append(cirq.measure(q0, q1, q2, q3, key='qubits'))

print(circuit)
print("Квант сэлгэлт хийж байна түр хүлээгээрэй...", end='\r')
time.sleep(1)

simulator = cirq.Simulator()
result = simulator.run(circuit, repetitions=1000000)

print("                                            ", end='\r')
print("Ажиллуулж дууслаа одоо үр дүнг харуулая:    ", end='\r')
time.sleep(3)
for sample in result.measurements['qubits'][:300]:
    print(sample, end='\r')
    time.sleep(0.03)
    print("                                             ", end='\r')

print("Дууслаа.", end='\r')

print(result.histogram(key='qubits'))
