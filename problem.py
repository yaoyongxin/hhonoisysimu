"""Hamiltonian, fixed ansatz in Quantinuum native gates, and measurement groups.

Conventions (same as adaptvqite, which produced data/ansatz.json): ansatz state = prod_k exp(-i theta_k/2 P_k) |ref>,
first label applied first; Pauli labels and ref bit strings have the leftmost character on the
highest qiskit qubit index (same as qiskit labels).
"""
import os
import json
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

HA2CM = 219474.6313705
NATIVE = ["rz", "r", "rzz"]
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_H = os.path.join(HERE, "data", "hpaulis.txt")
DEFAULT_ANSATZ = os.path.join(HERE, "data", "ansatz.json")
DEFAULT_EREF = os.path.join(HERE, "data", "eref")


def load_hamiltonian(path=DEFAULT_H, factor=HA2CM):
    """hpaulis.txt ('coef * PAULI' in Hartree) -> SparsePauliOp in cm^-1."""
    terms = []
    for line in open(path):
        if line.strip():
            c, s = line.split("*")
            terms.append((s.strip(), float(c)*factor))
    return SparsePauliOp.from_list(terms).simplify()


def load_ansatz(path=DEFAULT_ANSATZ):
    """JSON with "labels" (Pauli strings), "params" (angles), "ref_state" (bit string)."""
    d = json.load(open(path))
    return list(d["labels"]), np.array(d["params"], float), d["ref_state"]


def load_eref(path=DEFAULT_EREF):
    """reference eigenvalues (cm^-1) from lines 'E=<value> cm^-1, ...'."""
    out = []
    for line in open(path):
        if line.startswith("E="):
            out.append(float(line.split("=")[1].split()[0]))
    return np.array(out)


def _to_z(qc, q, p):
    """basis change V with V P V^dag = Z on qubit q (V applied first)."""
    if p == "X":
        qc.h(q)
    elif p == "Y":
        qc.sdg(q)
        qc.h(q)


def _from_z(qc, q, p):
    if p == "X":
        qc.h(q)
    elif p == "Y":
        qc.h(q)
        qc.s(q)


def pauli_rotation(qc, label, theta):
    """append exp(-i theta/2 P): weight 1 -> one U1q/Rz; weight 2 -> one RZZ(theta);
    weight w > 2 -> CX ladder (w-2 each side, each CX = one ZZMax) around one RZZ(theta)."""
    n = qc.num_qubits
    sup = [(n - 1 - i, p) for i, p in enumerate(label) if p != "I"]
    if len(sup) == 1:
        q, p = sup[0]
        {"X": qc.rx, "Y": qc.ry, "Z": qc.rz}[p](theta, q)
        return
    for q, p in sup:
        _to_z(qc, q, p)
    qs = [q for q, _ in sup]
    for a, b in zip(qs[:-2], qs[1:-1]):
        qc.cx(a, b)
    qc.rzz(theta, qs[-2], qs[-1])
    for a, b in reversed(list(zip(qs[:-2], qs[1:-1]))):
        qc.cx(a, b)
    for q, p in sup:
        _from_z(qc, q, p)


def build_ansatz(labels, ref="000000", native=True):
    n = len(labels[0])
    th = ParameterVector("theta", len(labels))
    qc = QuantumCircuit(n)
    for i, b in enumerate(reversed(ref)):
        if b == "1":
            qc.x(i)
    for t, lab in zip(th, labels):
        pauli_rotation(qc, lab, t)
    if native:
        # level 1: basis translation (CX -> ZZMax = RZZ(pi/2) + 1q) and 1q merging only
        qc = transpile(qc, basis_gates=NATIVE, optimization_level=1, seed_transpiler=0)
    return qc, th


class MeasGroup:
    """One qubit-wise commuting group: native basis-change circuit and parity table."""

    def __init__(self, op: SparsePauliOp):
        n = op.num_qubits
        basis = ["I"]*n                                  # per qiskit qubit
        for lab in op.paulis.to_labels():
            for i, p in enumerate(reversed(lab)):
                if p != "I":
                    basis[i] = p
        tail = QuantumCircuit(n)
        for q, p in enumerate(basis):
            _to_z(tail, q, p)
        self.tail = transpile(tail, basis_gates=NATIVE, optimization_level=1)
        self.basis = basis
        self.coeffs = op.coeffs.real.copy()
        idx = np.arange(2**n)
        signs = []
        for lab in op.paulis.to_labels():
            mask = sum(1 << i for i, p in enumerate(reversed(lab)) if p != "I")
            par = np.array([bin(x & mask).count("1") & 1 for x in idx])
            signs.append(1 - 2*par)
        self.signs = np.array(signs, float)              # (nterms, 2^n)

    def energy(self, probs):
        return float(self.coeffs @ (self.signs @ probs))


def measurement_groups(H: SparsePauliOp):
    ident = "I"*H.num_qubits
    const = float(sum(c.real for p, c in zip(H.paulis.to_labels(), H.coeffs) if p == ident))
    rest = SparsePauliOp.from_list([(p, c) for p, c in zip(H.paulis.to_labels(), H.coeffs)
                                    if p != ident])
    return const, [MeasGroup(g) for g in rest.group_commuting(qubit_wise=True)]
