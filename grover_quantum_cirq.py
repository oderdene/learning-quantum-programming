import cirq


def oracle_00():
    q0, q1  = cirq.LineQubit.range(2)

    print("|00> oracle")
    circuit = cirq.Circuit()
    circuit.append(cirq.H(q0))
    circuit.append(cirq.H(q1))
    circuit.append(cirq.X(q0))
    circuit.append(cirq.X(q1))
    circuit.append(cirq.CZ(q0, q1))
    circuit.append(cirq.X(q0))
    circuit.append(cirq.X(q1))

    # amplitude amplification
    circuit.append(cirq.H(q0))
    circuit.append(cirq.H(q1))
    circuit.append(cirq.Z(q0))
    circuit.append(cirq.Z(q1))
    circuit.append(cirq.CZ(q0, q1))
    circuit.append(cirq.H(q0))
    circuit.append(cirq.H(q1))

    circuit.append(cirq.measure(q0, q1, key='result'))

    print(circuit)

    sim    = cirq.Simulator()
    result = sim.run(circuit, repetitions=1000)
    print(result.histogram(key='result'))

    sample = result.measurements['result'][0]
    q0 = sample[0]
    q1 = sample[1]
    print("result : |{}{}>".format(q0, q1))


if __name__=="__main__":
    oracle_00()

