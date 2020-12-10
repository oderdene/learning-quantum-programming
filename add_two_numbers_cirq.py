import sys
import cirq

print("Бүхэл тоо оруулна уу!")

a = int(input("a="))
b = int(input("b="))

n_bits = 8
a_bits = '{:08b}'.format(a)[-n_bits:]
b_bits = '{:08b}'.format(b)[-n_bits:]



def sum_qubits(a_bit, b_bit, carry_in):
    q0, q1, q2, q3, q4, q5, q6, q7 = cirq.LineQubit.range(8)
    # q0 qubit-ийг а битд зориулж хэрэглэнэ
    # q1 qubit-ийг б битд зориулж хэрэглэнэ
    # q2 qubit-ийг carry bit-д зориулна
    def assign_bit(bit, qubit):
        if bit==1:
            yield cirq.X(qubit)

    circuit = cirq.Circuit(
            assign_bit(a_bit, q0),
            assign_bit(b_bit, q1),
            assign_bit(carry_in, q2),
            # AND1
            cirq.TOFFOLI(q0, q1, q3),
            # XOR1
            cirq.CNOT(q0, q4),
            cirq.CNOT(q1, q4),
            # XOR2
            cirq.CNOT(q2, q5),
            cirq.CNOT(q4, q5),
            # AND2
            cirq.TOFFOLI(q2, q4, q6),
            # OR
            cirq.X(q3),
            cirq.X(q6),
            cirq.TOFFOLI(q3, q6, q7),
            cirq.X(q7), # carry out
            cirq.measure(q5, q7, key='results')
            )
    simulator = cirq.Simulator()
    result    = simulator.run(circuit, repetitions=10).measurements['results'][0]
    return result[0], result[1]


print("Full adder бит нэмэх хүснэгт:")

a_bit, b_bit, carry_in = 1, 1, 1
summed, carry_out = sum_qubits(a_bit=a_bit, b_bit=b_bit, carry_in=carry_in)
print("a={} b={} carry_in={} sum={} carry_out={}".format(a_bit, b_bit, carry_in, summed, carry_out))

a_bit, b_bit, carry_in = 0, 1, 1
summed, carry_out = sum_qubits(a_bit=a_bit, b_bit=b_bit, carry_in=carry_in)
print("a={} b={} carry_in={} sum={} carry_out={}".format(a_bit, b_bit, carry_in, summed, carry_out))

a_bit, b_bit, carry_in = 1, 0, 1
summed, carry_out = sum_qubits(a_bit=a_bit, b_bit=b_bit, carry_in=carry_in)
print("a={} b={} carry_in={} sum={} carry_out={}".format(a_bit, b_bit, carry_in, summed, carry_out))

a_bit, b_bit, carry_in = 0, 0, 1
summed, carry_out = sum_qubits(a_bit=a_bit, b_bit=b_bit, carry_in=carry_in)
print("a={} b={} carry_in={} sum={} carry_out={}".format(a_bit, b_bit, carry_in, summed, carry_out))

a_bit, b_bit, carry_in = 1, 1, 0
summed, carry_out = sum_qubits(a_bit=a_bit, b_bit=b_bit, carry_in=carry_in)
print("a={} b={} carry_in={} sum={} carry_out={}".format(a_bit, b_bit, carry_in, summed, carry_out))

a_bit, b_bit, carry_in = 0, 1, 0
summed, carry_out = sum_qubits(a_bit=a_bit, b_bit=b_bit, carry_in=carry_in)
print("a={} b={} carry_in={} sum={} carry_out={}".format(a_bit, b_bit, carry_in, summed, carry_out))

a_bit, b_bit, carry_in = 1, 0, 0
summed, carry_out = sum_qubits(a_bit=a_bit, b_bit=b_bit, carry_in=carry_in)
print("a={} b={} carry_in={} sum={} carry_out={}".format(a_bit, b_bit, carry_in, summed, carry_out))

a_bit, b_bit, carry_in = 0, 0, 0
summed, carry_out = sum_qubits(a_bit=a_bit, b_bit=b_bit, carry_in=carry_in)
print("a={} b={} carry_in={} sum={} carry_out={}".format(a_bit, b_bit, carry_in, summed, carry_out))


print("########### ХОЁР БҮХЭЛ ТООНЫ НИЙЛБЭР ##############")
result = ''
carry_in = 0
for i in reversed(range(n_bits)):
    a_bit    = int(a_bits[i])
    b_bit    = int(b_bits[i])
    summed, carry_out = sum_qubits(a_bit=a_bit, b_bit=b_bit, carry_in=carry_in)
    carry_in = carry_out
    result  += str(summed)
result     = result[::-1]
result_int = int(result, 2)

print("{}={} \n{}={}".format(a, a_bits, b, b_bits))
print("{}+{}={}".format(a_bits, b_bits, result))
print("{}={}".format(result, result_int))
print("########### ҮР ДҮН #################################")
print("{}+{}={}".format(a, b, result_int))
