# hhonoisysimu

Noisy simulation of a variational quantum eigensolver (VQE) for a 6-qubit anharmonic
vibrational Hamiltonian (HHO), using a fixed, compact ansatz and a noise model that follows
Quantinuum's H2 trapped-ion emulator. The aim is to study how device noise and shot noise
carry through to VQE energies and parameters, as a testbed for error-control methods that will
be added later.

The code switches between four settings: noiseless, shot noise only, device noise only (infinite
shots, exact) and device + shot noise. A global factor scales the device noise, and each noise
channel can be switched on or off separately for error budgets.

## Contents

| path | content |
|---|---|
| `data/hpaulis.txt` | Hamiltonian as a sum of Pauli strings, `coef * PAULI`, coefficients in Hartree (216 terms, 6 qubits) |
| `data/eref` | reference lowest eigenvalues and excitation energies (cm^-1) |
| `data/ansatz.json` | fixed ansatz: 45 Pauli-rotation labels, their angles, reference state `000000` |
| `problem.py` | loads the Hamiltonian (converted to cm^-1), the ansatz, builds the native-gate circuit and the measurement groups |
| `quantinuum_noise.py` | `QuantinuumNoise` (H2 emulator noise model, presets, scaling) and `add_noise()` |
| `estimator.py` | `NoisyEstimator`: energy estimates in the four modes, optional purification |
| `run_vqe.py` | VQE driver (SPSA or COBYLA), writes a JSON with the trace and diagnostics |
| `scan_noise.py` | error budget at the stored angles: energy vs noise scale, and per channel |
| `check_setup.py` | sanity checks (spectrum vs `eref`, circuit synthesis, noiseless limits) |
| `run_baseline.sbatch` | Slurm script for a baseline set of runs over all modes |

## Installation

Python >= 3.10 with

```bash
pip install -r requirements.txt     # qiskit >= 2.0, qiskit-aer >= 0.17, numpy, scipy
python check_setup.py
```

Developed with qiskit 2.6.2, qiskit-aer 0.18.2, numpy 2.5, scipy 1.18 (Python 3.12).
`check_setup.py` should print the five eigenvalues of `data/eref`
(4653.2039, 6230.4050, 7791.1512, 8369.5471, 8435.8508 cm^-1), confirm the circuit synthesis,
and report the ansatz energy E - E0 = 0.0495 cm^-1 with ground-state fidelity 0.999994.

## The problem

- 6 qubits, 216 Pauli terms; the Hamiltonian is real (Y operators appear in pairs).
- Energies are reported in cm^-1 (coefficients x 219474.6313705). Ground state E0 = 4653.2039 cm^-1,
  first excitation 1577.2 cm^-1.
- Qubit-wise commuting grouping gives 26 measurement settings.

## The ansatz

`data/ansatz.json` comes from an adaptive variational imaginary-time evolution (AVQITE) run, kept
fixed here. The state is

    |psi(theta)> = prod_k exp(-i theta_k/2 P_k) |000000>,     first label applied first,

with label strings in qiskit order (leftmost character = highest qubit index). The 45 rotations
come from a qubit-excitation-type pool: X/Y strings on 2 qubits (XY, YX) and 4 qubits
(XXXY-, XYYY-type), plus single-qubit Y. The single-qubit Y rotations are needed: strings that flip
two or four qubits keep the parity of the bit string, while the ground state has 2.2% weight in
the odd-parity sector.

Composition: 15 one-qubit, 23 two-qubit and 7 four-qubit rotations.

### Native-gate circuit

Quantinuum's native gates are U1q (qiskit `r(theta, phi)`), virtual Rz, and the arbitrary-angle
ZZ gate RZZ(theta) = exp(-i theta/2 ZZ), with ZZMax = RZZ(pi/2). Each rotation is synthesized by hand
(`problem.pauli_rotation`):

- weight 1: one U1q (X, Y) or Rz (Z);
- weight 2: basis change to ZZ, one RZZ(theta), basis change back;
- weight 4: basis change, a CX ladder down to two qubits (2 CX on each side, each CX = one ZZMax
  plus 1-qubit gates), one RZZ(theta), ladder back.

After a level-1 qiskit transpilation (basis translation and 1-qubit merging only) the circuit has
58 RZZ (23 + 7 x 5), 213 U1q and 14 Rz, with 2-qubit depth 43. Generic qiskit transpilation of the
same rotations gives 82 RZZ.

## Noise model (`quantinuum_noise.py`)

Follows the noise model in the *Quantinuum System Model H2 Emulator Product Data Sheet* v1.2
(2024-06-04). Default parameters (preset `H2-1E-2025`) are the H2-1E emulator values listed in
Quantinuum's documentation as of 2025-07-16; preset `H2-1E-2024` holds the data-sheet values.

| parameter | H2-1E-2025 | H2-1E-2024 | meaning |
|---|---|---|---|
| `p1` | 1.89e-5 | 2.9e-5 | fault probability of a U1q gate |
| `p2` | 1.05e-3 | 1.28e-3 | fault probability of ZZMax = RZZ(pi/2) |
| `p_meas` | (6e-4, 1.39e-3) | (5e-4, 2.5e-3) | readout flip probability for outcome 0, 1 |
| `p_init` | 4e-5 | 4e-5 | initialization fault probability |
| `p1_emission_ratio` | 0.32 | 0.32 | fraction of p1 that is spontaneous emission |
| `p2_emission_ratio` | 0.48 | 0.48 | fraction of p2 that is emission, per qubit of a 2-qubit gate |
| `quadratic_dephasing_rate` | 0.043 | 0.043 | f_q, in units of 2 pi rad/s |
| `linear_dephasing_rate` | 0.0028 | 0.0028 | r_lin, 1/s |
| `coherent_to_incoherent_factor` | 2.0 | 2.0 | multiplier on f_q for the incoherent variant |
| `przz` (a, b, c, d, power) | (1.518, 0.241, 1.518, 0.241, 1) | same | arbitrary-angle RZZ scaling |

The 2025 page does not list emission ratios; the 2024 values are used.

Channels, as implemented:

- **U1q gate**: Pauli fault with probability p1 (1 - p1_emission_ratio), uniform over X, Y, Z;
  then spontaneous emission with probability p1 p1_emission_ratio.
- **RZZ(theta) gate**: two-qubit Pauli fault (uniform over the 15 non-identity Paulis) with
  probability p2 (1 - p2_emission_ratio) f(theta), where

      f = przz_a |theta/pi|^power + przz_b   (theta < 0)
      f = przz_c |theta/pi|^power + przz_d   (theta > 0)
      f = (przz_b + przz_d)/2                (theta = 0)

  with theta wrapped to [-pi, pi). This is normalized so that f(pi/2) = 1, and small-angle gates
  are less noisy (f -> 0.241). Then spontaneous emission on each of the two qubits with probability
  p2 p2_emission_ratio, which does not depend on the angle.
- **Spontaneous emission**: X with probability 1/4, Y with 1/4, leakage with 1/2.
- **Rz**: noiseless (virtual).
- **Memory (dephasing) noise**: per qubit, for a duration d: a coherent RZ(2 pi f_q d) and a Pauli Z
  with probability r_lin d. With `coherent_dephasing=False` (`--incoherent`) the coherent rotation
  is replaced by a Pauli Z with probability sin^2(pi f_q c d), c = coherent_to_incoherent_factor.
- **Initialization**: X flip with probability p_init on each qubit.
- **Readout**: per qubit, a 0 is read as 1 with p_meas[0] and a 1 as 0 with p_meas[1]; applied as a
  confusion matrix to the outcome distributions.

Choices not fixed by the data sheet (adjust as needed):

- The Pauli weights of the "asymmetric depolarizing" channels are not published. The default is
  uniform; `weights_1q` and `weights_2q` take custom weights.
- Leakage leaves the qubit space. Here a leaked qubit is replaced by the maximally mixed state
  (reset to 0 or 1 with probability 1/2 each).
- Timing: the circuit is divided into ASAP layers of parallel 2-qubit gates, and every qubit dephases
  for `t_layer` = 0.05 s per layer (all ions are transported every layer). The dephasing a qubit
  accumulates since its previous 2-qubit gate is inserted just before its next one. This t_layer
  gives a Z probability of about 1.9e-4 per qubit per layer, the order of the published H-series
  memory error per depth-1 circuit time. The coherent part adds up over the 43 layers, about
  0.58 rad of Z rotation per qubit at scale 1, so results depend on this choice; see the budget below.
- Crosstalk only happens during mid-circuit measurement and reset, which these circuits don't have,
  so it is not modelled.

### Scaling

`scale` multiplies all error probabilities: 0 is noiseless, 1 is hardware-like. The coherent
dephasing angle is multiplied by sqrt(scale), so its error probability sin^2(angle/2) also scales
about linearly. The per-channel factors `p1_scale`, `p2_scale`, `emission_scale`, `memory_scale`,
`meas_scale` and `init_scale` act on disjoint channels: p1/p2 scale the depolarizing parts only,
emission the spontaneous-emission parts of both gate types. `QuantinuumNoise.only("p2", ...)` gives
a copy with all other channels switched off.

## Estimator (`estimator.py`)

`NoisyEstimator(H, labels, ref, noise, shots, seed, gs, purify)` has four modes:

| mode | `noise` | `shots` | method |
|---|---|---|---|
| ideal | `None` | `None` | statevector, exact expectation values |
| shots | `None` | N | multinomial sampling of the ideal outcome distributions, N shots per group |
| noisy | model | `None` | exact expectation values from the noisy density matrix |
| noisy_shots | model | N | sampling of the noisy outcome distributions (readout errors included) |

Pipeline, one method per stage so error control can be inserted at any point:

1. `bind(theta)`: native circuit with angles bound.
2. `noisy_body(theta, noise)`: `add_noise()` inserts Kraus instructions after each gate, memory
   channels per layer, initialization faults.
3. `rho(theta)`: density matrix from qiskit-aer (`density_matrix` method); `purify` is applied here.
4. `exact_probs(theta)`: for each measurement group, the noisy 1-qubit basis-change channels are
   applied to rho (numpy), then the readout confusion; this gives the exact outcome distributions.
5. `sample(probs, shots)`: multinomial shot noise.
6. `energy(theta)`: sum over the groups plus the identity term.

Any of `shots`, `noise` and `purify` can be overridden in a call, e.g.
`est.energy(theta, shots=None, purify=None)` gives the exact noisy energy at the same parameters.
`exact_probs_aer()` is a slower reference path (one full Aer circuit per group). It agrees with the
fast path to 1e-14.

Speed: about 0.1 s per noisy energy evaluation (all 26 groups), and much less in the ideal modes.

### Purification (mode noisy only)

At 6 qubits the full density matrix (64 x 64) is cheap, so the infinite-shot noisy case is exact. It
also gives direct access to rho for purification-type error control:

- `vd2`, `vd3`: virtual distillation, rho -> rho^M / Tr rho^M;
- `dominant`: projector onto the dominant eigenvector of rho.

Purification acts on the prepared state; the measurement basis change and readout errors are
applied after it.

## Usage

```bash
# error budget at the stored angles
python scan_noise.py                       # coherent dephasing (emulator default)
python scan_noise.py --incoherent

# VQE
python run_vqe.py --mode ideal       --optimizer cobyla
python run_vqe.py --mode shots       --shots 10000 --optimizer spsa --maxiter 500
python run_vqe.py --mode noisy       --scale 0.5   --optimizer cobyla --maxiter 2000
python run_vqe.py --mode noisy       --scale 1     --purify vd2 --optimizer cobyla
python run_vqe.py --mode noisy_shots --scale 1     --shots 10000 --optimizer spsa --out a.json

# baseline set on a Slurm node (writes runs/*.json, runs/*.log)
PY=/path/to/env/bin/python sbatch -o $PWD/slurm-%j.out run_baseline.sbatch $PWD
```

Main options of `run_vqe.py`:

| option | default | meaning |
|---|---|---|
| `--mode` | `noisy_shots` | `ideal`, `shots`, `noisy`, `noisy_shots` |
| `--scale` | 1.0 | global device-noise scale |
| `--shots` | 10000 | shots per measurement group (26 groups) |
| `--purify` | none | `vd2`, `vd3`, `dominant` (mode `noisy` only) |
| `--preset` | `H2-1E-2025` | noise parameter set |
| `--t-layer` | 0.05 | seconds per 2-qubit layer (memory noise) |
| `--incoherent` | off | incoherent instead of coherent quadratic dephasing |
| `--optimizer` | `spsa` | `spsa` or `cobyla` |
| `--maxiter` | 300 | iterations |
| `--init` | `avqite` | stored angles, `zero` (reference state), or `perturb` (stored + Gaussian noise of width `--perturb`) |
| `--rhobeg` | 0.05 | COBYLA initial step |
| `--spsa-c`, `--spsa-step`, `--spsa-maxstep` | 0.05, 0.01, 0.1 | SPSA perturbation, calibrated first step, cap on the L2 norm of an update |
| `--diag-every` | 10 | iterations between diagnostics |
| `--hamiltonian`, `--ansatz` | `data/...` | input files |

SPSA uses the standard gain sequences (alpha 0.602, gamma 0.101, A = 0.1 maxiter), with the gain
calibrated from 10 gradient samples at the start point. The update cap matters: started at a
(near) stationary point with little noise, the calibration sees a small gradient and gives a huge
gain. Without the cap a shot-noise-only run diverged to E - E0 ~ 11000 cm^-1.

### Output

`run_vqe.py` writes a JSON with the arguments, mode, noise parameters, circuit counts, E0 and the
first excitations, the optimizer trace (estimated energy per evaluation), and diagnostics every
`--diag-every` iterations and at the start and end. Each diagnostic holds the parameters and:

- `E_est`: the estimate the optimizer used (with shot noise if any);
- `E_exact`: the run's own mode at infinite shots (after purification, if any);
- `E_noisy`: the device model at `--scale`, infinite shots, no purification. This is recorded in
  every mode, so parameters optimized without noise can be checked on the noisy device;
- `E_ideal`: noiseless energy;
- `fidelity`: |<GS|psi_ideal(theta)>|^2.

This splits the error into

    E_est - E0 = (E_ideal - E0)       ansatz and optimization
               + (E_exact - E_ideal)  device-noise bias at these parameters
               + (E_est - E_exact)    shot noise

## Results so far

### Error budget at the stored angles (`scan_noise.py`, scale 1, cm^-1)

| channel | E - E_ideal, coherent dephasing | E - E_ideal, incoherent dephasing |
|---|---|---|
| p1 (1-qubit depolarizing) | 12.8 | 12.8 |
| p2 (2-qubit depolarizing) | 139.5 | 139.5 |
| emission (incl. leakage) | 260.3 | 260.3 |
| memory | 916.8 | 359.0 |
| readout | 16.5 | 16.5 |
| initialization | 1.1 | 1.1 |
| all | 1315.5 | 771.2 |

The bias grows about linearly with the global scale: 137, 340, 672, 1316, 2518, 4611 cm^-1 at scales
0.1, 0.25, 0.5, 1, 2, 4 (coherent). The purity of the noisy state at scale 1 is 0.865.

### Baseline VQE runs (from the stored angles; energies minus E0, cm^-1)

| mode | optimizer | E_est | E_exact | E_noisy | E_ideal | fidelity |
|---|---|---|---|---|---|---|
| ideal | COBYLA | 0.05 | 0.05 | 1316 | 0.05 | 0.999994 |
| shots, 10^4/group | SPSA, 500 it | 91 | 79 | 1668 | 79 | 0.990 |
| noisy, scale 0.25 | COBYLA, 2000 it | 216 | 216 | 216 | 124 | 0.983 |
| noisy, scale 0.5 | COBYLA, 2000 it | 427 | 427 | 427 | 248 | 0.965 |
| noisy, scale 1 | COBYLA, 2000 it | 829 | 829 | 829 | 503 | 0.930 |
| noisy, scale 1, vd2 | COBYLA, 2000 it | 287 | 287 | 830 | 489 | 0.932 |
| noisy, scale 1, dominant | COBYLA, 2000 it | 282 | 282 | 830 | 492 | 0.931 |
| noisy_shots, scale 0.25, 10^4 | SPSA, 500 it | 251 | 225 | 225 | 128 | 0.983 |
| noisy_shots, scale 0.5, 10^4 | SPSA, 500 it | 430 | 436 | 436 | 263 | 0.963 |
| noisy_shots, scale 1, 10^4 | SPSA, 500 it | 783 | 848 | 848 | 509 | 0.932 |

Observations:

- Optimizing against device noise lowers the noisy energy (1316 -> 829 cm^-1 at scale 1) by moving
  the parameters away from the true ground state, toward angles where the noise does less damage.
  The ideal energy of the optimized angles rises to ~500 cm^-1 above E0 (fidelity 0.93). Both the
  bias and the parameter shift grow about linearly with the scale.
- Purification lowers the reported energy (to ~285 cm^-1 at scale 1) but finds equally shifted
  parameters (E_ideal ~490, fidelity 0.93): the coherent part of the noise survives purification,
  and the optimizer still compensates for it.
- Shot noise alone at 10^4 shots per group gives E_ideal ~80 cm^-1 above E0 (fidelity 0.990); one
  energy estimate at this budget has a standard deviation of ~50 cm^-1.
- The noisy COBYLA runs used all 2000 iterations and may not be fully converged.

## Adding error control

- Circuit level (Pauli twirling, dynamical decoupling, gate folding): transform the bound circuit
  between `bind` and `add_noise` (`NoisyEstimator.noisy_body`).
- Zero-noise extrapolation: call `energy(theta, noise=...)` with copies of the model at several
  `scale` values (`dataclasses.replace(noise, scale=s, _cache={})`).
- Readout mitigation: act on the distributions returned by `exact_probs` / `sample`.
- State-level methods (purification, subspace expansion): work on `rho(theta)`.

## References

- Quantinuum System Model H2 Emulator Product Data Sheet, v1.2 (2024-06-04).
- Quantinuum documentation, System Model H2 Emulators (H2-1E noise parameters, 2025-07-16).
- Quantinuum System Model H2 Product Data Sheet.
