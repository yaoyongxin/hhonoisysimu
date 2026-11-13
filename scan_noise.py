"""Error budget of the fixed ansatz at its stored parameters (no optimization):
exact noisy energy vs global noise scale, and per noise channel at scale 1.

usage: python scan_noise.py [--preset H2-1E-2025] [--scales 0 0.1 ...] [--out scan.json]
"""
import argparse
import json
import numpy as np

from problem import load_hamiltonian, load_ansatz, DEFAULT_ANSATZ, DEFAULT_H
from estimator import NoisyEstimator
from quantinuum_noise import QuantinuumNoise, CHANNELS

ap = argparse.ArgumentParser()
ap.add_argument("--hamiltonian", default=DEFAULT_H, help="Pauli sum in Hartree")
ap.add_argument("--ansatz", default=DEFAULT_ANSATZ, help="ansatz JSON")
ap.add_argument("--preset", default="H2-1E-2025")
ap.add_argument("--t-layer", type=float, default=0.05)
ap.add_argument("--incoherent", action="store_true", help="incoherent quadratic dephasing")
ap.add_argument("--scales", type=float, nargs="+", default=[0, 0.1, 0.25, 0.5, 1, 2, 4])
ap.add_argument("--out", default="scan_noise.json")
args = ap.parse_args()

H = load_hamiltonian(args.hamiltonian)
labels, theta, ref = load_ansatz(args.ansatz)
w, v = np.linalg.eigh(H.to_matrix())
base = QuantinuumNoise.preset(args.preset, t_layer=args.t_layer,
                              coherent_dephasing=not args.incoherent)
est = NoisyEstimator(H, labels, ref, base, gs=v[:, 0])
e_id = est.ideal_energy(theta)
print(f"circuit: {est.counts()}")
print(f"E0 = {w[0]:.4f} cm^-1, ansatz ideal E - E0 = {e_id - w[0]:.4f}, "
      f"fidelity = {est.fidelity(theta):.6f}")
print(f"memory per qubit per layer: coherent angle, Z prob = {base.memory_per_layer()}")

res = {"preset": args.preset, "noise": base.to_dict(), "E0": w[0], "E_ideal": e_id,
       "counts": est.counts(), "scales": [], "channels": {}}
print("\nglobal scale   E_noisy - E0   E_noisy - E_ideal   purity")
for s in args.scales:
    nz = QuantinuumNoise.preset(args.preset, t_layer=args.t_layer, scale=s,
                                coherent_dephasing=not args.incoherent)
    e = est.energy(theta, shots=None, noise=nz)
    rho = est.rho(theta, nz, None)
    pur = float(np.real(np.trace(rho @ rho)))
    res["scales"].append({"scale": s, "E": e, "dE": e - e_id, "purity": pur})
    print(f"{s:10.3g}   {e - w[0]:12.3f}   {e - e_id:16.3f}   {pur:.4f}")

print("\nchannel (scale 1, others off)   E_noisy - E_ideal (cm^-1)")
for ch in CHANNELS:
    e = est.energy(theta, shots=None, noise=base.only(ch))
    res["channels"][ch] = e - e_id
    print(f"  {ch:28s} {e - e_id:10.3f}")
tot = est.energy(theta, shots=None, noise=base) - e_id
print(f"  {'all':28s} {tot:10.3f}   (sum of channels {sum(res['channels'].values()):.3f})")
res["channels"]["all"] = tot
json.dump(res, open(args.out, "w"), indent=1)
print(f"wrote {args.out}")
