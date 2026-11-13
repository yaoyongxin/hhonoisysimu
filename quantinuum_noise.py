"""Quantinuum H2-like noise model for qiskit-aer.

Follows the H2 emulator noise model described in the Quantinuum System Model H2 Emulator
Product Data Sheet v1.2 (2024-06-04):

* 1-qubit gate (U1q = qiskit ``r``): fault with probability p1. A fraction p1_emission_ratio of
  it is spontaneous emission, the rest (asymmetric) depolarizing Pauli noise.
* 2-qubit gate (RZZ(theta), ZZMax = RZZ(pi/2)): depolarizing fault probability
  p2 * f(theta) * (1 - p2_emission_ratio), with the arbitrary-angle scaling
      f = przz_a |theta/pi|^przz_power + przz_b   (theta < 0)
      f = przz_c |theta/pi|^przz_power + przz_d   (theta > 0)
      f = (przz_b + przz_d)/2                     (theta = 0),
  normalized so that f(pi/2) = 1; plus spontaneous emission on each qubit with probability
  p2 * p2_emission_ratio (same for every angle).
* Spontaneous emission: X with 1/4, Y with 1/4, leakage with 1/2. Leakage is outside the qubit
  space; here it is approximated by resetting the qubit to the maximally mixed state.
* Rz is virtual (noiseless).
* Memory (dephasing) noise during transport and idling, per qubit and per duration d:
  coherent RZ(2 pi f_q d) with f_q = quadratic_dephasing_rate, plus Pauli Z with probability
  linear_dephasing_rate * d. With coherent_dephasing=False the quadratic part is Pauli Z with
  probability sin^2(pi f_q c d), c = coherent_to_incoherent_factor.
* Initialization: X flip with probability p_init. Measurement: bit flip with p_meas[0] on a
  0 outcome and p_meas[1] on a 1 outcome (applied by the estimator as a confusion matrix).

Choices made here that the data sheet does not fix (documented, adjustable):
* The Pauli weights of the "asymmetric" depolarizing channels are not published; default uniform.
* Durations: the circuit is cut into layers of parallel 2-qubit gates (ASAP); every qubit
  dephases for t_layer per layer (transport happens for all ions every layer). t_layer = 0.05 s
  gives a Z probability ~1.9e-4 per qubit per layer, the order of the published H-series
  memory error per depth-1 circuit time.
* Crosstalk (only for mid-circuit measurement/reset) is not modelled: these circuits have none.

Scaling: ``scale`` multiplies all error probabilities (0: noiseless, 1: hardware-like). The
coherent dephasing angle is multiplied by sqrt(scale * memory_scale), so that its error
probability sin^2(angle/2) also scales ~linearly. Per-channel scales follow the emulator, except
that the channels are disjoint (for error budgets): p1_scale / p2_scale scale only the
depolarizing parts, emission_scale the spontaneous-emission parts of both gate types.
"""
from dataclasses import dataclass, field, asdict, replace
import itertools
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import RZGate
from qiskit.quantum_info import Kraus, Pauli

# Default emulator parameters. 2025 values: H2-1E docs page as of 2025-07-16 (p1, p2, p_meas,
# p_init, dephasing, przz); emission ratios not listed there, taken from the 2024 data sheet.
PRESETS = {
    "H2-1E-2025": dict(p1=1.89e-5, p2=1.05e-3, p_meas=(6e-4, 1.39e-3), p_init=4e-5,
                       p1_emission_ratio=0.32, p2_emission_ratio=0.48,
                       quadratic_dephasing_rate=0.043, linear_dephasing_rate=0.0028,
                       coherent_to_incoherent_factor=2.0,
                       przz=(1.518, 0.241, 1.518, 0.241, 1.0)),
    "H2-1E-2024": dict(p1=2.9e-5, p2=1.28e-3, p_meas=(5e-4, 2.5e-3), p_init=4e-5,
                       p1_emission_ratio=0.32, p2_emission_ratio=0.48,
                       quadratic_dephasing_rate=0.043, linear_dephasing_rate=0.0028,
                       coherent_to_incoherent_factor=2.0,
                       przz=(1.518, 0.241, 1.518, 0.241, 1.0)),
}

CHANNELS = ("p1", "p2", "emission", "memory", "meas", "init")


@dataclass
class QuantinuumNoise:
    p1: float = 1.89e-5
    p2: float = 1.05e-3
    p_meas: tuple = (6e-4, 1.39e-3)
    p_init: float = 4e-5
    p1_emission_ratio: float = 0.32
    p2_emission_ratio: float = 0.48
    quadratic_dephasing_rate: float = 0.043   # Hz (units of 2 pi rad/s)
    linear_dephasing_rate: float = 0.0028     # 1/s
    coherent_to_incoherent_factor: float = 2.0
    coherent_dephasing: bool = True
    t_layer: float = 0.05                     # s per 2-qubit-gate layer (assumption, see above)
    przz: tuple = (1.518, 0.241, 1.518, 0.241, 1.0)
    weights_1q: dict = None                   # e.g. {"X": .25, "Y": .25, "Z": .5}; None: uniform
    weights_2q: dict = None                   # 15 two-qubit Paulis -> weight; None: uniform
    scale: float = 1.0
    p1_scale: float = 1.0
    p2_scale: float = 1.0
    emission_scale: float = 1.0
    memory_scale: float = 1.0
    meas_scale: float = 1.0
    init_scale: float = 1.0
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def preset(cls, name="H2-1E-2025", **kw):
        return cls(**{**PRESETS[name], **kw})

    def only(self, *channels):
        """Copy with every channel not listed switched off (for error budgets)."""
        kw = {f"{c}_scale": (getattr(self, f"{c}_scale") if c in channels else 0.0)
              for c in CHANNELS}
        return replace(self, _cache={}, **kw)

    def to_dict(self):
        d = asdict(self)
        d.pop("_cache")
        return d

    # ---- probabilities -------------------------------------------------------------------
    def przz_factor(self, theta):
        a, b, c, d, pw = self.przz
        theta = (theta + np.pi) % (2*np.pi) - np.pi      # wrap to [-pi, pi)
        x = abs(theta)/np.pi
        if theta < 0:
            return a*x**pw + b
        if theta > 0:
            return c*x**pw + d
        return 0.5*(b + d)

    def _p1_dep(self):
        return self.p1*(1 - self.p1_emission_ratio)*self.scale*self.p1_scale

    def _p1_em(self):
        return self.p1*self.p1_emission_ratio*self.scale*self.emission_scale

    def _p2_dep(self, theta):
        return (self.p2*(1 - self.p2_emission_ratio)*self.przz_factor(theta)
                * self.scale*self.p2_scale)

    def _p2_em(self):
        return self.p2*self.p2_emission_ratio*self.scale*self.emission_scale

    def meas_flip(self):
        s = self.scale*self.meas_scale
        return self.p_meas[0]*s, self.p_meas[1]*s

    def memory_per_layer(self):
        """(coherent RZ angle, incoherent Z probability) per qubit per layer."""
        s = self.scale*self.memory_scale
        d = self.t_layer
        pz = self.linear_dephasing_rate*d*s
        if self.coherent_dephasing:
            ang = 2*np.pi*self.quadratic_dephasing_rate*d*np.sqrt(s)
        else:
            ang = 0.0
            pq = np.sin(np.pi*self.quadratic_dephasing_rate
                        * self.coherent_to_incoherent_factor*d)**2*s
            pz = pz + pq - 2*pz*pq
        return ang, pz

    # ---- channels (Kraus instructions; cached) ------------------------------------------
    @staticmethod
    def _pauli_channel(probs):
        """probs: {pauli label (qiskit order): p}; identity gets the rest."""
        n = len(next(iter(probs)))
        ptot = sum(probs.values())
        ops = [np.sqrt(1 - ptot)*np.eye(2**n)]
        ops += [np.sqrt(p)*Pauli(k).to_matrix() for k, p in probs.items() if p > 0]
        return Kraus(ops).to_instruction()

    @staticmethod
    def _emission(pe):
        """X 1/4, Y 1/4, leakage 1/2 (approximated by reset to the maximally mixed state)."""
        k = np.sqrt(pe/4)
        e = np.eye(2)
        ops = [np.sqrt(1 - pe)*e, k*Pauli("X").to_matrix(), k*Pauli("Y").to_matrix()]
        ops += [k*np.outer(e[i], e[j]) for i in range(2) for j in range(2)]
        return Kraus(ops).to_instruction()

    def channels_1q(self):
        """list of Kraus instructions applied after a U1q gate."""
        if "1q" not in self._cache:
            p, pe = self._p1_dep(), self._p1_em()
            w = self.weights_1q or {"X": 1/3, "Y": 1/3, "Z": 1/3}
            tot = sum(w.values())
            out = []
            if p > 0:
                out.append(self._pauli_channel({k: p*v/tot for k, v in w.items()}))
            if pe > 0:
                out.append(self._emission(pe))
            self._cache["1q"] = out
        return self._cache["1q"]

    def channels_2q(self, theta):
        """list of (Kraus instruction, local qubits) applied after RZZ(theta)."""
        key = ("2q", round(float(theta), 12))
        if key not in self._cache:
            p, pe = self._p2_dep(theta), self._p2_em()
            out = []
            if p > 0:
                if self.weights_2q is None:
                    labels = ["".join(t) for t in itertools.product("IXYZ", repeat=2)
                              if t != ("I", "I")]
                    w = {k: 1.0 for k in labels}
                else:
                    w = self.weights_2q
                tot = sum(w.values())
                out.append((self._pauli_channel({k: p*v/tot for k, v in w.items()}), [0, 1]))
            if pe > 0:
                em = self._emission(pe)
                out += [(em, [0]), (em, [1])]
            self._cache[key] = out
        return self._cache[key]

    def channels_memory(self, nlayers):
        """list of instructions (Z channel, coherent RZ) for nlayers layers on one qubit."""
        key = ("mem", nlayers)
        if key not in self._cache:
            ang, pz = self.memory_per_layer()
            out = []
            if nlayers > 0:
                pz_n = 0.5*(1 - (1 - 2*pz)**nlayers)
                if pz_n > 0:
                    out.append(self._pauli_channel({"Z": pz_n}))
                if ang != 0:
                    out.append(RZGate(ang*nlayers))
            self._cache[key] = out
        return self._cache[key]

    def channels_init(self):
        p = self.p_init*self.scale*self.init_scale
        return [self._pauli_channel({"X": p})] if p > 0 else []


def add_noise(circ: QuantumCircuit, noise: QuantinuumNoise, init=True, final_memory=True,
              layer_offset=None):
    """Return a copy of a bound native circuit (rz, r, rzz) with noise instructions inserted.

    Memory noise: each qubit dephases for t_layer per layer of parallel 2-qubit gates; the
    dephasing accumulated since the qubit's previous 2-qubit gate is inserted just before its
    next one (and at the end of the circuit for the remaining layers).
    layer_offset: per-qubit layer counters to continue from (for circuits appended later).
    Returns (noisy circuit, final per-qubit layer counters, total layers).
    """
    n = circ.num_qubits
    out = QuantumCircuit(n)
    if init:
        for ch in noise.channels_init():
            for q in range(n):
                out.append(ch, [q])
    ch1 = noise.channels_1q()
    level = [0]*n if layer_offset is None else list(layer_offset)  # layer of last 2q gate
    done = list(level)                                          # layers of memory applied

    def memory(q, upto):
        k = upto - done[q]
        if k > 0:
            for ch in noise.channels_memory(k):
                out.append(ch, [q])
            done[q] = upto

    for inst in circ.data:
        op = inst.operation
        qs = [circ.find_bit(x).index for x in inst.qubits]
        name = op.name
        if name == "barrier":
            continue
        if name == "rzz":
            L = max(level[q] for q in qs) + 1
            for q in qs:
                memory(q, L)
                level[q] = L
            out.append(op, qs)
            for ch, loc in noise.channels_2q(float(op.params[0])):
                out.append(ch, [qs[i] for i in loc])
        elif name == "r":
            out.append(op, qs)
            for ch in ch1:
                out.append(ch, qs)
        elif name == "rz":
            out.append(op, qs)
        else:
            raise ValueError(f"non-native gate {name}; transpile to rz, r, rzz first")
    total = max(level)
    if final_memory:
        for q in range(n):
            memory(q, total)
    return out, level, total
