"""Energy estimator with switchable device noise and shot noise.

Four settings (``noise`` x ``shots``):
    noise=None,  shots=None : ideal (statevector, exact expectation values)
    noise=None,  shots=N    : shot noise only (sampling from the ideal distributions)
    noise=model, shots=None : device noise only, infinite shots (exact noisy expectation values
                              from the density matrix; optional purification of rho)
    noise=model, shots=N    : device noise + shot noise

Pipeline (each stage is a method, so error-control steps can be inserted later):
    bind(theta) -> native circuit
    noisy_body(theta) -> noisy state-preparation circuit                  [noise model]
    rho(theta) -> its density matrix (Aer, density_matrix method), optionally purified
    exact_probs(theta) -> per group: (noisy) basis change, readout error -> exact distributions
    sample(probs) -> shot-noise estimate of the distributions
    energy(theta) -> sum over groups (+ identity constant)

Purification (only with device noise and infinite shots, since it needs rho itself):
    "vd2", "vd3": virtual distillation rho^M / Tr rho^M
    "dominant": projector on the dominant eigenvector of rho
It acts on the prepared state; the measurement basis change and readout errors are applied after.
"""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Kraus
from qiskit_aer import AerSimulator

from problem import measurement_groups, build_ansatz
from quantinuum_noise import QuantinuumNoise, add_noise

DEFAULT = object()
PURIFY = (None, "vd2", "vd3", "dominant")


def purify_rho(rho, method):
    if method is None:
        return rho
    if method in ("vd2", "vd3"):
        m = rho @ rho if method == "vd2" else rho @ rho @ rho
        return m/np.trace(m).real
    if method == "dominant":
        w, v = np.linalg.eigh(rho)
        return np.outer(v[:, -1], v[:, -1].conj())
    raise ValueError(f"unknown purification {method}")


class NoisyEstimator:
    def __init__(self, H, labels, ref="000000", noise: QuantinuumNoise = None, shots=None,
                 seed=None, gs=None, purify=None):
        self.H = H
        self.n = H.num_qubits
        self.circ, self.theta = build_ansatz(labels, ref)
        self.const, self.groups = measurement_groups(H)
        self.noise = noise                      # None: no device noise
        self.shots = shots                      # per group; None/0: exact expectation values
        if purify not in PURIFY:
            raise ValueError(f"purify must be one of {PURIFY}")
        if purify and (noise is None or shots):
            raise ValueError("purification needs device noise and infinite shots (shots=None)")
        self.purify = purify
        self.rng = np.random.default_rng(seed)
        self.sim = AerSimulator(method="density_matrix")
        self.gs = gs                            # exact ground state (for fidelity), optional
        self.neval = 0
        self.nshots_total = 0
        self._tail = {}

    @property
    def mode(self):
        return {(False, False): "ideal", (False, True): "shots",
                (True, False): "noisy", (True, True): "noisy_shots"}[
            (self.noise is not None, bool(self.shots))]

    # ---- circuit info --------------------------------------------------------------------
    def counts(self):
        ops = self.circ.count_ops()
        twoq = self.circ.depth(lambda g: g.operation.num_qubits == 2)
        return {"ops": dict(ops), "depth_2q": twoq, "groups": len(self.groups)}

    def bind(self, theta):
        return self.circ.assign_parameters(dict(zip(self.theta, theta)))

    # ---- noisy simulation ----------------------------------------------------------------
    def noisy_body(self, theta, noise):
        """noisy ansatz circuit (state preparation), without measurement basis change."""
        body, _, _ = add_noise(self.bind(theta), noise)
        return body

    def _tail_kraus(self, noise):
        """per (qubit, basis) Kraus operators of the (noisy) basis-change circuit (1q only)."""
        key = id(noise)
        if key not in self._tail:
            tail = {}
            for g in self.groups:
                for q, b in enumerate(g.basis):
                    if (q, b) in tail:
                        continue
                    c1 = QuantumCircuit(1)
                    for i in g.tail.data:
                        if g.tail.find_bit(i.qubits[0]).index == q:
                            c1.append(i.operation, [0])
                    if noise is not None:
                        c1, _, _ = add_noise(c1, noise, init=False, final_memory=False)
                    tail[(q, b)] = np.array(Kraus(c1).data)
            self._tail[key] = tail
        return self._tail[key]

    def rho(self, theta, noise=DEFAULT, purify=DEFAULT):
        """density matrix of the prepared state (pure if noise is None)."""
        noise = self.noise if noise is DEFAULT else noise
        purify = self.purify if purify is DEFAULT else purify
        if noise is None:
            psi = Statevector(self.bind(theta)).data
            return np.outer(psi, psi.conj())
        c = self.noisy_body(theta, noise)
        c.save_density_matrix()
        r = np.asarray(self.sim.run(c).result().data(0)["density_matrix"])
        return purify_rho(r, purify)

    def _probs_from_rho(self, rho, noise):
        n = self.n
        tails = self._tail_kraus(noise)
        out = []
        for g in self.groups:
            t = rho.reshape((2,)*(2*n))
            for q, b in enumerate(g.basis):
                ks = tails[(q, b)]
                r, c = n - 1 - q, 2*n - 1 - q          # row / column axes of qubit q
                acc = 0
                for k in ks:
                    tk = np.moveaxis(np.tensordot(k, t, axes=([1], [r])), 0, r)
                    tk = np.moveaxis(np.tensordot(k.conj(), tk, axes=([1], [c])), 0, c)
                    acc = acc + tk
                t = acc
            p = np.real(np.diagonal(t.reshape(2**n, 2**n))).copy()
            out.append(self.readout(np.clip(p, 0, None), noise))
        return out

    def _probs_from_psi(self, psi):
        out = []
        for g in self.groups:
            out.append(Statevector(psi).evolve(g.tail).probabilities())
        return out

    def readout(self, probs, noise):
        if noise is None:
            return probs
        p0, p1 = noise.meas_flip()
        if p0 == 0 and p1 == 0:
            return probs
        M = np.array([[1 - p0, p1], [p0, 1 - p1]])   # M[measured, true]
        t = probs.reshape((2,)*self.n)
        for ax in range(self.n):
            t = np.moveaxis(np.tensordot(M, t, axes=([1], [ax])), 0, ax)
        return t.reshape(-1)

    def exact_probs(self, theta, noise=DEFAULT, purify=DEFAULT):
        noise = self.noise if noise is DEFAULT else noise
        if noise is None:
            return self._probs_from_psi(Statevector(self.bind(theta)))
        return self._probs_from_rho(self.rho(theta, noise, purify), noise)

    def exact_probs_aer(self, theta, noise=DEFAULT):
        """reference path: one full Aer circuit per group (slow; for validation)."""
        noise = self.noise if noise is DEFAULT else noise
        body, level, _ = add_noise(self.bind(theta), noise)
        circs = []
        for g in self.groups:
            tail, _, _ = add_noise(g.tail, noise, init=False, final_memory=False,
                                   layer_offset=level)
            c = body.compose(tail)
            c.save_probabilities()
            circs.append(c)
        res = self.sim.run(circs).result()
        return [self.readout(np.clip(np.asarray(res.data(i)["probabilities"]), 0, None), noise)
                for i in range(len(self.groups))]

    def sample(self, probs, shots):
        out = []
        for p in probs:
            p = p/p.sum()
            out.append(self.rng.multinomial(shots, p)/shots)
            self.nshots_total += shots
        return out

    def energy(self, theta, shots=DEFAULT, noise=DEFAULT, purify=DEFAULT):
        """estimate of <H>; defaults to the estimator's own mode, any part can be overridden
        (e.g. shots=None, purify=None for the exact noisy energy at the same parameters)."""
        shots = self.shots if shots is DEFAULT else shots
        probs = self.exact_probs(theta, noise, purify)
        if shots:
            probs = self.sample(probs, shots)
        self.neval += 1
        return self.const + sum(g.energy(p) for g, p in zip(self.groups, probs))

    # ---- noiseless references ------------------------------------------------------------
    def ideal_state(self, theta):
        return Statevector(self.bind(theta))

    def ideal_energy(self, theta):
        return float(self.ideal_state(theta).expectation_value(self.H).real)

    def fidelity(self, theta):
        if self.gs is None:
            return np.nan
        return float(abs(np.vdot(self.gs, self.ideal_state(theta).data))**2)
