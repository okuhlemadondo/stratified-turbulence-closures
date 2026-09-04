"""
audit_experiments.py

Ablation experiments responding to the coherence audit.
Runs the 2×2 diagnostic matrix [Raw / Normalized Basis] × [Adam / Plain GD]
plus a cold-start baseline and a frozen-coefficient ablation.

The goal: determine whether Path B's rebound is caused by:
  (a) Gram conditioning / basis scaling (→ vanishes under normalized basis)
  (b) Adam momentum overshoot (→ vanishes under plain GD)
  (c) Genuine penalty-barrier collision (→ persists in both)
"""

import os
os.environ['MPLCONFIGDIR'] = os.path.abspath('.cache/matplotlib')

import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from prototype_square_duct import SquareDuctQuadrant, AdamOptimizer, check_realizability
from curvature_experiment import ExtendedDuctMesh

np.random.seed(42)


# ==============================================================================
# Plain Gradient Descent Optimizer
# ==============================================================================
class PlainGD:
    """Plain gradient descent with fixed learning rate."""
    def __init__(self, lr=2e-3):
        self.lr = lr

    def step(self, theta, grad):
        return theta - self.lr * grad


# ==============================================================================
# Generic Path Runner
# ==============================================================================
def run_path(mesh, basis, tau_ref, W, norm_ref, theta_init, n_steps,
             lambda_realiz=150.0, optimizer_cls=AdamOptimizer, lr=2e-3,
             freeze_mask=None):
    """
    Run optimization for n_steps. Returns history dict.

    Parameters
    ----------
    basis : list of ndarray, the tensor basis [T1, T2, ...]
    theta_init : initial parameter vector
    freeze_mask : boolean array, True = freeze that coefficient
    """
    d = len(basis)
    theta = theta_init.copy()
    I3 = np.eye(3)[None, None, :, :]

    opt = optimizer_cls(lr=lr)

    history = {"loss": [], "viol": [], "theta": []}

    for step in range(n_steps):
        # Predict
        tau_dev = sum(theta[i] * basis[i] for i in range(d))
        tau = (2.0 / 3.0) * mesh.k[:, :, None, None] * I3 + tau_dev

        # Task loss + gradient
        diff = tau - tau_ref
        loss_task = 0.5 * np.sum(W * (diff ** 2))
        grad_task = np.array([np.sum(W * diff * basis[i]) for i in range(d)])

        # Realizability penalty + gradient
        loss_realiz = 0.0
        grad_realiz = np.zeros(d)
        if lambda_realiz > 0:
            tau_sym = 0.5 * (tau + np.swapaxes(tau, 2, 3))
            eigvals, eigvecs = np.linalg.eigh(tau_sym)
            min_eig = eigvals[:, :, 0]
            min_vec = eigvecs[:, :, :, 0]
            violation_mask = (min_eig < 0)
            if np.any(violation_mask):
                viol = np.where(violation_mask, min_eig, 0.0)
                loss_realiz = lambda_realiz * np.sum(mesh.W_grid * (viol ** 2))
                for i in range(d):
                    v_T_v = np.einsum('abi,abij,abj->ab', min_vec, basis[i], min_vec)
                    grad_realiz[i] = 2.0 * lambda_realiz * np.sum(mesh.W_grid * viol * v_T_v)

        total_loss = loss_task + loss_realiz
        total_grad = grad_task + grad_realiz

        # Record
        viol_frac = 1.0 - check_realizability(mesh, tau)
        history["loss"].append(float(total_loss))
        history["viol"].append(float(viol_frac))
        history["theta"].append(theta.tolist())

        # Apply freeze mask
        if freeze_mask is not None:
            total_grad = total_grad.copy()
            total_grad[freeze_mask] = 0.0

        # Step
        theta = opt.step(theta, total_grad)

    return history, theta


# ==============================================================================
# Main Audit Experiments
# ==============================================================================
def main():
    print("=" * 80)
    print("COHERENCE AUDIT — ABLATION EXPERIMENTS")
    print("=" * 80)

    mesh = ExtendedDuctMesh(h=1.0, Ny=48, Nz=48)
    W = mesh.W_grid[:, :, None, None]
    norm_ref = np.sqrt(np.sum(W * (mesh.tau_DNS ** 2)))

    raw_basis = [mesh.T1, mesh.T2, mesh.T3, mesh.T4]

    # ======================================================================
    # 0. FULL GRAM MATRIX DIAGNOSTICS
    # ======================================================================
    print("\n" + "=" * 80)
    print("0. GRAM MATRIX DIAGNOSTICS")
    print("=" * 80)

    G = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            G[i, j] = np.sum(W * raw_basis[i] * raw_basis[j])

    print("\nFull Gram Matrix G:")
    for i in range(4):
        print(f"  [{', '.join(f'{G[i, j]:+.6e}' for j in range(4))}]")

    # Normalized correlation matrix
    D = np.sqrt(np.diag(G))
    R = G / np.outer(D, D)
    print("\nNormalized Correlation Matrix R:")
    for i in range(4):
        print(f"  [{', '.join(f'{R[i, j]:+.6f}' for j in range(4))}]")

    # L² norms
    norms = [float(np.sqrt(np.sum(W * T ** 2))) for T in raw_basis]
    print(f"\nL² norms: {[f'{n:.6e}' for n in norms]}")

    # Block condition numbers
    G_01 = G[:2, :2]
    G_23 = G[2:, 2:]
    kappa_01 = float(np.linalg.cond(G_01))
    kappa_23 = float(np.linalg.cond(G_23))
    eigs_full = np.linalg.eigvalsh(G)
    kappa_full = float(eigs_full[-1] / eigs_full[0])
    print(f"\nBlock κ(θ₀,θ₁) = {kappa_01:.2f}")
    print(f"Block κ(θ₂,θ₃) = {kappa_23:.2f}")
    print(f"Full  κ(G)      = {kappa_full:.2f}")

    # Unit-normalized Gram
    G_norm = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            G_norm[i, j] = G[i, j] / (norms[i] * norms[j])
    eigs_norm = np.linalg.eigvalsh(G_norm)
    kappa_norm = float(eigs_norm[-1] / eigs_norm[0])
    print(f"\nUnit-normalized κ(G̃) = {kappa_norm:.2f}")

    # ======================================================================
    # 1. BASE CALIBRATION (Stratum 0)
    # ======================================================================
    print("\n" + "=" * 80)
    print("1. BASE STRATUM 0 CALIBRATION")
    print("=" * 80)

    _, theta_0_star = run_path(
        mesh, [mesh.T1], mesh.tau_DNS, W, norm_ref,
        theta_init=np.array([0.01]),
        n_steps=120, lambda_realiz=0.0,
        optimizer_cls=AdamOptimizer, lr=2e-3
    )
    print(f"θ₀* = {theta_0_star[0]:.6f}")

    # Prepare unit-normalized basis
    norm_basis = [T / n for T, n in zip(raw_basis, norms)]

    # For normalized basis, we need to transform theta_0_star to normalized coordinates
    # In raw basis: τ_dev = θ₀ T₁. In normalized: τ_dev = θ̃₀ T̃₁ where T̃₁ = T₁/‖T₁‖
    # So θ̃₀ = θ₀ · ‖T₁‖
    theta_0_star_norm = np.array([theta_0_star[0] * norms[0]])

    # ======================================================================
    # 2. THE 2×2 DIAGNOSTIC MATRIX + COLD START + FROZEN
    # ======================================================================
    lam = 150.0
    N_total = 220
    N_phase1 = 100
    N_phase2 = 120

    results = {}

    conditions = {
        "raw_adam": {
            "basis": raw_basis, "opt_cls": AdamOptimizer,
            "theta0_base": theta_0_star[0], "label": "Raw Basis + Adam"
        },
        "norm_adam": {
            "basis": norm_basis, "opt_cls": AdamOptimizer,
            "theta0_base": theta_0_star_norm[0], "label": "Normalized Basis + Adam"
        },
        "raw_gd": {
            "basis": raw_basis, "opt_cls": PlainGD,
            "theta0_base": theta_0_star[0], "label": "Raw Basis + Plain GD"
        },
        "norm_gd": {
            "basis": norm_basis, "opt_cls": PlainGD,
            "theta0_base": theta_0_star_norm[0], "label": "Normalized Basis + Plain GD"
        },
    }

    for cond_name, cfg in conditions.items():
        print(f"\n{'=' * 80}")
        print(f"CONDITION: {cfg['label']}")
        print(f"{'=' * 80}")
        basis = cfg["basis"]
        opt_cls = cfg["opt_cls"]
        t0 = cfg["theta0_base"]

        # --- Path A: Sequential (100 in Stratum 1, 120 in Stratum 2) ---
        theta_init_s1 = np.array([t0, 0.0, 0.0])
        hist_A1, theta_1_star = run_path(
            mesh, basis[:3], mesh.tau_DNS, W, norm_ref,
            theta_init=theta_init_s1,
            n_steps=N_phase1, lambda_realiz=lam,
            optimizer_cls=opt_cls, lr=2e-3
        )

        theta_init_s2a = np.array([theta_1_star[0], theta_1_star[1], theta_1_star[2], 0.0])
        hist_A2, theta_2A_star = run_path(
            mesh, basis, mesh.tau_DNS, W, norm_ref,
            theta_init=theta_init_s2a,
            n_steps=N_phase2, lambda_realiz=lam,
            optimizer_cls=opt_cls, lr=2e-3
        )

        hist_A = {
            "loss": hist_A1["loss"] + hist_A2["loss"],
            "viol": hist_A1["viol"] + hist_A2["viol"],
        }

        # --- Path B: Direct (220 in Stratum 2) ---
        theta_init_s2b = np.array([t0, 0.0, 0.0, 0.0])
        hist_B, theta_2B_star = run_path(
            mesh, basis, mesh.tau_DNS, W, norm_ref,
            theta_init=theta_init_s2b,
            n_steps=N_total, lambda_realiz=lam,
            optimizer_cls=opt_cls, lr=2e-3
        )

        # Metrics
        max_viol_A = max(hist_A["viol"])
        max_viol_B = max(hist_B["viol"])
        # Detect rebound: look for loss increase > 10% after initial descent in Path B
        losses_B = hist_B["loss"]
        rebound_ratio = 1.0
        for i in range(10, min(80, len(losses_B))):
            window_min = min(losses_B[max(0, i - 10):i])
            if window_min > 0 and losses_B[i] / window_min > rebound_ratio:
                rebound_ratio = losses_B[i] / window_min

        final_loss_A = hist_A["loss"][-1]
        final_loss_B = hist_B["loss"][-1]
        final_viol_A = hist_A["viol"][-1]
        final_viol_B = hist_B["viol"][-1]

        print(f"  Path A: max viol = {max_viol_A * 100:.2f}%, final loss = {final_loss_A:.4e}, final viol = {final_viol_A * 100:.2f}%")
        print(f"  Path B: max viol = {max_viol_B * 100:.2f}%, final loss = {final_loss_B:.4e}, final viol = {final_viol_B * 100:.2f}%")
        print(f"  Path B rebound ratio = {rebound_ratio:.3f}x")

        results[cond_name] = {
            "label": cfg["label"],
            "path_A": {"loss": hist_A["loss"], "viol": hist_A["viol"],
                       "max_viol": max_viol_A, "final_loss": final_loss_A, "final_viol": final_viol_A},
            "path_B": {"loss": hist_B["loss"], "viol": hist_B["viol"],
                       "max_viol": max_viol_B, "final_loss": final_loss_B, "final_viol": final_viol_B,
                       "rebound_ratio": rebound_ratio},
        }

    # ======================================================================
    # 3. COLD-START BASELINE (Path C)
    # ======================================================================
    print(f"\n{'=' * 80}")
    print("EXPERIMENT A: COLD-START BASELINE (Path C)")
    print(f"{'=' * 80}")

    hist_C, theta_C_star = run_path(
        mesh, raw_basis, mesh.tau_DNS, W, norm_ref,
        theta_init=np.array([0.0, 0.0, 0.0, 0.0]),
        n_steps=N_total, lambda_realiz=lam,
        optimizer_cls=AdamOptimizer, lr=2e-3
    )
    print(f"  Path C: max viol = {max(hist_C['viol']) * 100:.2f}%, "
          f"final loss = {hist_C['loss'][-1]:.4e}, final viol = {hist_C['viol'][-1] * 100:.2f}%")

    results["cold_start"] = {
        "label": "Cold Start (θ=[0,0,0,0])",
        "path_C": {"loss": hist_C["loss"], "viol": hist_C["viol"],
                   "max_viol": max(hist_C["viol"]),
                   "final_loss": hist_C["loss"][-1],
                   "final_viol": hist_C["viol"][-1]},
    }

    # ======================================================================
    # 4. FROZEN-COEFFICIENT ABLATION (Path A, θ₀..θ₂ frozen in phase 2)
    # ======================================================================
    print(f"\n{'=' * 80}")
    print("EXPERIMENT D: FROZEN-COEFFICIENT ABLATION")
    print(f"{'=' * 80}")

    # Phase 1: same as raw_adam Path A phase 1
    theta_init_s1 = np.array([theta_0_star[0], 0.0, 0.0])
    hist_D1, theta_1_star_D = run_path(
        mesh, raw_basis[:3], mesh.tau_DNS, W, norm_ref,
        theta_init=theta_init_s1,
        n_steps=N_phase1, lambda_realiz=lam,
        optimizer_cls=AdamOptimizer, lr=2e-3
    )

    # Phase 2: freeze θ₀, θ₁, θ₂ — only optimize θ₃
    theta_init_s2d = np.array([theta_1_star_D[0], theta_1_star_D[1], theta_1_star_D[2], 0.0])
    freeze = np.array([True, True, True, False])
    hist_D2, theta_D_star = run_path(
        mesh, raw_basis, mesh.tau_DNS, W, norm_ref,
        theta_init=theta_init_s2d,
        n_steps=N_phase2, lambda_realiz=lam,
        optimizer_cls=AdamOptimizer, lr=2e-3,
        freeze_mask=freeze
    )

    hist_D = {
        "loss": hist_D1["loss"] + hist_D2["loss"],
        "viol": hist_D1["viol"] + hist_D2["viol"],
    }

    print(f"  Frozen Path A: max viol = {max(hist_D['viol']) * 100:.2f}%, "
          f"final loss = {hist_D['loss'][-1]:.4e}, final viol = {hist_D['viol'][-1] * 100:.2f}%")
    print(f"  Final θ (frozen): {theta_D_star}")

    results["frozen"] = {
        "label": "Frozen θ₀..θ₂ in Phase 2",
        "path_D": {"loss": hist_D["loss"], "viol": hist_D["viol"],
                   "max_viol": max(hist_D["viol"]),
                   "final_loss": hist_D["loss"][-1],
                   "final_viol": hist_D["viol"][-1],
                   "final_theta": theta_D_star.tolist()},
    }

    # ======================================================================
    # 5. SUMMARY TABLE
    # ======================================================================
    print(f"\n{'=' * 80}")
    print("2×2 DIAGNOSTIC MATRIX SUMMARY")
    print(f"{'=' * 80}")
    print(f"{'Condition':<30s} | {'B max viol%':>12s} | {'B rebound':>10s} | {'B final loss':>14s} | {'A final loss':>14s}")
    print("-" * 90)
    for cond_name in ["raw_adam", "norm_adam", "raw_gd", "norm_gd"]:
        r = results[cond_name]
        print(f"{r['label']:<30s} | {r['path_B']['max_viol'] * 100:>11.2f}% | "
              f"{r['path_B']['rebound_ratio']:>9.3f}x | "
              f"{r['path_B']['final_loss']:>14.4e} | "
              f"{r['path_A']['final_loss']:>14.4e}")

    print(f"\n{'Additional Experiments':<30s} | {'Max viol%':>12s} | {'Final loss':>14s}")
    print("-" * 60)
    print(f"{'Cold Start (Path C)':<30s} | {results['cold_start']['path_C']['max_viol'] * 100:>11.2f}% | "
          f"{results['cold_start']['path_C']['final_loss']:>14.4e}")
    print(f"{'Frozen θ₀..θ₂ (Path D)':<30s} | {results['frozen']['path_D']['max_viol'] * 100:>11.2f}% | "
          f"{results['frozen']['path_D']['final_loss']:>14.4e}")

    # ======================================================================
    # 6. SAVE RESULTS
    # ======================================================================
    # Serialize (convert numpy types)
    def sanitize(obj):
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [sanitize(x) for x in obj]
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    gram_data = {
        "G_full": G.tolist(),
        "R_correlation": R.tolist(),
        "L2_norms": norms,
        "kappa_block_01": kappa_01,
        "kappa_block_23": kappa_23,
        "kappa_full": kappa_full,
        "kappa_unit_normalized": kappa_norm,
        "eigenvalues_raw": eigs_full.tolist(),
        "eigenvalues_normalized": eigs_norm.tolist(),
    }

    output = {
        "gram_diagnostics": gram_data,
        "experiments": sanitize(results),
    }

    with open("results_audit_experiments.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to results_audit_experiments.json")

    # ======================================================================
    # 7. DIAGNOSTIC PLOTS
    # ======================================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Panel (0,0): 2×2 matrix — Path B Loss
    ax = axes[0, 0]
    ax.set_title("Path B Loss Trajectories\n(2×2 Matrix)", fontsize=10)
    for cond_name in ["raw_adam", "norm_adam", "raw_gd", "norm_gd"]:
        r = results[cond_name]
        ax.semilogy(r["path_B"]["loss"], label=r["label"], alpha=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=7)

    # Panel (0,1): 2×2 matrix — Path B Violation
    ax = axes[0, 1]
    ax.set_title("Path B Violation %\n(2×2 Matrix)", fontsize=10)
    for cond_name in ["raw_adam", "norm_adam", "raw_gd", "norm_gd"]:
        r = results[cond_name]
        ax.plot([v * 100 for v in r["path_B"]["viol"]], label=r["label"], alpha=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Violation %")
    ax.legend(fontsize=7)

    # Panel (0,2): Cold Start vs Warm Start
    ax = axes[0, 2]
    ax.set_title("Cold Start (Path C) vs\nWarm Starts (A, B)", fontsize=10)
    r_ra = results["raw_adam"]
    ax.semilogy(r_ra["path_A"]["loss"], label="Path A (Sequential)", alpha=0.8)
    ax.semilogy(r_ra["path_B"]["loss"], label="Path B (Direct)", alpha=0.8)
    ax.semilogy(results["cold_start"]["path_C"]["loss"], label="Path C (Cold Start)", alpha=0.8, ls="--")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=8)

    # Panel (1,0): 2×2 matrix — Path A Loss
    ax = axes[1, 0]
    ax.set_title("Path A Loss Trajectories\n(2×2 Matrix)", fontsize=10)
    for cond_name in ["raw_adam", "norm_adam", "raw_gd", "norm_gd"]:
        r = results[cond_name]
        ax.semilogy(r["path_A"]["loss"], label=r["label"], alpha=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=7)

    # Panel (1,1): Frozen vs Unfrozen
    ax = axes[1, 1]
    ax.set_title("Frozen θ₀..θ₂ (Path D)\nvs Unfrozen Path A", fontsize=10)
    ax.semilogy(r_ra["path_A"]["loss"], label="Path A (all θ free)", alpha=0.8)
    ax.semilogy(results["frozen"]["path_D"]["loss"], label="Path D (θ₀..θ₂ frozen)", alpha=0.8, ls="--")
    ax.axvline(x=100, color='gray', ls=':', alpha=0.5, label="Hop e₂")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=8)

    # Panel (1,2): Gram conditioning
    ax = axes[1, 2]
    ax.set_title("Gram Matrix Conditioning:\nRaw vs Unit-Normalized", fontsize=10)
    strata_labels = ["Str. 0\n(d=1)", "Str. 1\n(d=3)", "Str. 2\n(d=4)"]
    kappas_raw = [1.0, kappa_01, kappa_full]
    kappas_norm_list = [1.0]
    # Compute normalized κ for Stratum 1
    G1_norm = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            G1_norm[i, j] = G[i, j] / (norms[i] * norms[j])
    eigs_G1_norm = np.linalg.eigvalsh(G1_norm)
    kappas_norm_list.append(float(eigs_G1_norm[-1] / eigs_G1_norm[0]))
    kappas_norm_list.append(kappa_norm)

    x = np.arange(3)
    w = 0.35
    ax.bar(x - w / 2, kappas_raw, w, label="Raw basis", color="steelblue")
    ax.bar(x + w / 2, kappas_norm_list, w, label="Unit-normalized", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(strata_labels)
    ax.set_ylabel("κ(G)")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    for i in range(3):
        ax.text(x[i] - w / 2, kappas_raw[i] * 1.3, f"{kappas_raw[i]:.1f}", ha='center', fontsize=7)
        ax.text(x[i] + w / 2, kappas_norm_list[i] * 1.3, f"{kappas_norm_list[i]:.1f}", ha='center', fontsize=7)

    plt.tight_layout()
    plt.savefig("audit_ablation_results.png", dpi=200, bbox_inches='tight')
    print("Figure saved to audit_ablation_results.png")
    plt.close()


if __name__ == "__main__":
    main()
