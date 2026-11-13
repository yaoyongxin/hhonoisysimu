"""VQE with a fixed ansatz under a Quantinuum H2-like noise model and/or shot noise.

usage examples:
  python run_vqe.py --mode noisy_shots --scale 1 --shots 10000 --optimizer spsa --out a.json
  python run_vqe.py --mode noisy --scale 0.5 --optimizer cobyla --init zero
  python run_vqe.py --mode noisy --purify vd2 --optimizer cobyla

--mode:
  ideal        no device noise, exact expectation values (statevector)
  shots        shot noise only: --shots per measurement group (26 groups), no device noise
  noisy        device noise only, infinite shots: exact expectation values from the noisy
               density matrix; optional --purify vd2|vd3|dominant applied to rho
  noisy_shots  device noise (--scale) and shot noise (--shots)
The output JSON holds the configuration, the noise parameters, the optimizer trace
(estimated energy per evaluation) and diagnostics every --diag-every iterations:
ideal (noiseless) energy, exact noisy energy and ground-state fidelity of the current
parameters. This separates the final error into
    E_est - E0 = (E_ideal - E0)            [ansatz/optimization]
               + (E_exact - E_ideal)       [device-noise bias at these parameters,
                                            after purification if any]
               + (E_est - E_exact)         [shot noise]
E_noisy (device model, no purification) is recorded in every mode, so runs optimized without
noise can be checked against the noisy device at their parameters.
"""
import argparse
import json
import time
import numpy as np
from scipy.optimize import minimize

from problem import load_hamiltonian, load_ansatz, DEFAULT_ANSATZ, DEFAULT_H
from estimator import NoisyEstimator
from quantinuum_noise import QuantinuumNoise, PRESETS


def spsa(f, x0, maxiter, rng, c=0.05, alpha=0.602, gamma=0.101, target_step=0.01,
         max_step=0.1, ncal=10, callback=None):
    """Standard SPSA (Spall) with step calibration from ncal gradient samples at x0, and the
    L2 norm of each update capped at max_step (rad). The cap matters when x0 is (near) a
    stationary point: the calibration then sees a small gradient, gives a large gain, and
    the uncapped iteration runs away once it leaves x0."""
    x = np.array(x0, float)
    A = 0.1*maxiter
    g = []
    for _ in range(ncal):
        d = rng.choice([-1.0, 1.0], size=x.size)
        g.append(abs(f(x + c*d) - f(x - c*d))/(2*c))
    a = target_step*(A + 1)**alpha/max(np.mean(g), 1e-12)
    for k in range(maxiter):
        ak = a/(k + 1 + A)**alpha
        ck = c/(k + 1)**gamma
        d = rng.choice([-1.0, 1.0], size=x.size)
        fp, fm = f(x + ck*d), f(x - ck*d)
        step = ak*(fp - fm)/(2*ck)*d
        nrm = np.linalg.norm(step)
        if max_step and nrm > max_step:
            step *= max_step/nrm
        x = x - step
        if callback:
            callback(k, x, 0.5*(fp + fm))
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hamiltonian", default=DEFAULT_H, help="Pauli sum in Hartree")
    ap.add_argument("--ansatz", default=DEFAULT_ANSATZ, help="ansatz JSON (labels, params, ref)")
    ap.add_argument("--preset", default="H2-1E-2025", choices=list(PRESETS))
    ap.add_argument("--mode", default="noisy_shots",
                    choices=["ideal", "shots", "noisy", "noisy_shots"])
    ap.add_argument("--purify", default=None, choices=["vd2", "vd3", "dominant"],
                    help="purify the noisy density matrix (mode noisy only)")
    ap.add_argument("--scale", type=float, default=1.0, help="global device-noise scale")
    ap.add_argument("--t-layer", type=float, default=0.05, help="s per 2-qubit layer")
    ap.add_argument("--incoherent", action="store_true", help="incoherent quadratic dephasing")
    ap.add_argument("--shots", type=int, default=10000,
                    help="shots per group (modes shots, noisy_shots)")
    ap.add_argument("--optimizer", default="spsa", choices=["spsa", "cobyla"])
    ap.add_argument("--maxiter", type=int, default=300)
    ap.add_argument("--init", default="avqite", choices=["avqite", "zero", "perturb"],
                    help="start from the stored (AVQITE) angles, zeros (reference state), "
                         "or stored angles + Gaussian noise of width --perturb")
    ap.add_argument("--perturb", type=float, default=0.1)
    ap.add_argument("--rhobeg", type=float, default=0.05, help="COBYLA initial step")
    ap.add_argument("--spsa-c", type=float, default=0.05)
    ap.add_argument("--spsa-step", type=float, default=0.01,
                    help="calibrated size of the first SPSA step per parameter")
    ap.add_argument("--spsa-maxstep", type=float, default=0.1,
                    help="cap on the L2 norm of an SPSA update (0: none)")
    ap.add_argument("--diag-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="vqe.json")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    H = load_hamiltonian(args.hamiltonian)
    labels, th_avq, ref = load_ansatz(args.ansatz)
    w, v = np.linalg.eigh(H.to_matrix())
    device = QuantinuumNoise.preset(args.preset, scale=args.scale, t_layer=args.t_layer,
                                    coherent_dephasing=not args.incoherent)
    noisy = args.mode in ("noisy", "noisy_shots")
    noise = device if noisy else None
    shots = args.shots if args.mode in ("shots", "noisy_shots") else None
    if args.purify and args.mode != "noisy":
        ap.error("--purify needs --mode noisy (it acts on the exact density matrix)")
    est = NoisyEstimator(H, labels, ref, noise, shots=shots, seed=args.seed, gs=v[:, 0],
                         purify=args.purify)
    if args.init == "avqite":
        x0 = th_avq.copy()
    elif args.init == "zero":
        x0 = np.zeros_like(th_avq)
    else:
        x0 = th_avq + args.perturb*rng.normal(size=th_avq.size)

    def diag(x):
        """E_ideal: no noise; E_noisy: device model (always evaluated, even in modes ideal and
        shots, to see the noise bias at these parameters), infinite shots, no purification;
        E_exact: the run's own mode at infinite shots (= E_noisy unless purified/no noise)."""
        d = {"E_ideal": est.ideal_energy(x),
             "E_noisy": est.energy(x, shots=None, noise=device, purify=None),
             "E_exact": est.energy(x, shots=None),
             "fidelity": est.fidelity(x)}
        return d

    print(f"circuit {est.counts()}; E0 = {w[0]:.4f} cm^-1; mode {est.mode}, "
          f"scale {args.scale if noisy else 0}, shots/group {shots}, purify {args.purify}, "
          f"{args.optimizer}", flush=True)
    d0 = diag(x0)
    print(f"init: E_ideal-E0 {d0['E_ideal']-w[0]:.3f}  E_noisy-E0 {d0['E_noisy']-w[0]:.3f}  "
          f"E_exact-E0 {d0['E_exact']-w[0]:.3f}  fid {d0['fidelity']:.6f}", flush=True)

    trace = {"neval": [], "E_est": []}
    diags = []
    t0 = time.time()

    def f(x):
        e = est.energy(x)
        trace["neval"].append(est.neval)
        trace["E_est"].append(e)
        return e

    def record(it, x, e_est):
        if it % args.diag_every == 0 or it == args.maxiter - 1:
            d = diag(x)
            d.update(iter=it, neval=est.neval, E_est=e_est, theta=list(map(float, x)))
            diags.append(d)
            print(f"iter {it:5d} neval {est.neval:6d}  E_est-E0 {e_est-w[0]:10.3f}  "
                  f"E_exact-E0 {d['E_exact']-w[0]:9.3f}  E_noisy-E0 {d['E_noisy']-w[0]:9.3f}  "
                  f"E_ideal-E0 {d['E_ideal']-w[0]:8.3f}  "
                  f"fid {d['fidelity']:.6f}  {time.time()-t0:7.1f} s", flush=True)

    if args.optimizer == "spsa":
        x = spsa(f, x0, args.maxiter, rng, c=args.spsa_c, target_step=args.spsa_step,
                 max_step=args.spsa_maxstep, callback=record)
    else:
        it = [0]

        def cb(xk):
            record(it[0], xk, trace["E_est"][-1])
            it[0] += 1
        r = minimize(f, x0, method="COBYLA", callback=cb,
                     options={"maxiter": args.maxiter, "rhobeg": args.rhobeg})
        x = r.x

    dfin = diag(x)
    e_fin_est = est.energy(x)
    out = {"args": vars(args), "mode": est.mode, "noise": device.to_dict(),
           "noise_on": noisy, "counts": est.counts(),
           "E0": float(w[0]), "E_excited": list(map(float, w[1:5])),
           "init": d0, "final": {**dfin, "E_est": e_fin_est, "theta": list(map(float, x))},
           "neval": est.neval, "shots_total": est.nshots_total, "wall_s": time.time() - t0,
           "trace": trace, "diagnostics": diags}
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"final: E_est-E0 {e_fin_est-w[0]:.3f}  E_exact-E0 {dfin['E_exact']-w[0]:.3f}  "
          f"E_noisy-E0 {dfin['E_noisy']-w[0]:.3f}  "
          f"E_ideal-E0 {dfin['E_ideal']-w[0]:.3f}  fid {dfin['fidelity']:.6f}  "
          f"neval {est.neval}  shots {est.nshots_total}  -> {args.out}")


if __name__ == "__main__":
    main()
