"""Sanity checks of the inputs and the circuit:
1. exact diagonalization of data/hpaulis.txt reproduces data/eref;
2. the native-gate circuit equals the product of Pauli rotations (random angles);
3. the noiseless circuit at the stored angles gives the stored ansatz energy;
4. the noisy estimator at scale 0 equals the ideal energy.
usage: python check_setup.py
"""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import Pauli, Operator

from problem import load_hamiltonian, load_ansatz, load_eref, build_ansatz
from estimator import NoisyEstimator
from quantinuum_noise import QuantinuumNoise

H = load_hamiltonian()
labels, theta, ref = load_ansatz()
w, v = np.linalg.eigh(H.to_matrix())
eref = load_eref()
print("ED (cm^-1):   ", np.round(w[:len(eref)], 4))
print("eref (cm^-1): ", eref)
assert np.allclose(w[:len(eref)], eref, atol=1e-3), "spectrum does not match eref"

x = np.random.default_rng(0).normal(size=len(labels))
circ, _ = build_ansatz(labels, ref)
ref_c = QuantumCircuit(len(ref))
for t, lab in zip(x, labels):
    ref_c.append(PauliEvolutionGate(Pauli(lab), time=t/2), range(len(ref)))
assert Operator(circ.assign_parameters(x)).equiv(Operator(ref_c)), "synthesis mismatch"
print("native circuit == product of Pauli rotations: ok")

est = NoisyEstimator(H, labels, ref, None, gs=v[:, 0])
print("circuit:", est.counts())
e_id = est.ideal_energy(theta)
print(f"ansatz: E - E0 = {e_id - w[0]:.4f} cm^-1, fidelity = {est.fidelity(theta):.6f}")
e0 = est.energy(theta, noise=QuantinuumNoise(scale=0.0))
assert abs(e0 - e_id) < 1e-6, "noisy estimator at scale 0 differs from ideal"
print("noisy estimator at scale 0 == ideal: ok")
