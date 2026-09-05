"""
round3_investigations.py

Investigates:
B2: Matched-loss B-restart (at step 18 where loss first drops below Path A hop loss 4.73e-9)
B6: Per-coordinate Adam under normalized basis (alpha_n = alpha * ||T^(n)||, eps = 1e-16 or eps_n = eps / ||T^(n)||)
B7: Adam beta1 = 0 to matched final loss <= 1.3412e-9
B8: Step-size sweep for Plain GD and Adam (Pareto plot: peak violation vs steps-to-tolerance)
"""

import os
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parent
DATA_DIR = REPO_ROOT / "data"
PAPER_DIR = REPO_ROOT / "paper"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.environ['MPLCONFIGDIR'] = str(REPO_ROOT / '.cache' / 'matplotlib')

import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from prototype_square_duct import SquareDuctQuadrant, AdamOptimizer, check_realizability
from curvature_experiment import ExtendedDuctMesh

np.random.seed(42)

# ==============================================================================
# Optimizers
# ==============================================================================
class PlainGD:
    def __init__(self, lr=2e-3):
        self.lr = lr
    def step(self, theta, grad):
        return theta - self.lr * grad

class AdamNoBeta1:
    """Adam with beta1 = 0."""
    def __init__(self, lr=2e-3, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta2 = beta2
        self.eps = eps
        self.v = None
        self.t = 0
    def step(self, theta, grad):
        self.t += 1
        if self.v is None:
            self.v = np.zeros_like(grad)
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (grad ** 2)
        v_hat = self.v / (1.0 - self.beta2 ** self.t)
        return theta - self.lr * grad / (np.sqrt(v_hat) + self.eps)

class AdamPerCoord:
    """Adam with per-coordinate learning rates alpha_n and per-coordinate eps_n."""
    def __init__(self, lr_vec, beta1=0.9, beta2=0.999, eps_vec=None):
        self.lr_vec = np.array(lr_vec, dtype=float)
        self.beta1 = beta1
        self.beta2 = beta2
        if eps_vec is None:
            self.eps_vec = np.full_like(self.lr_vec, 1e-16)
        else:
            self.eps_vec = np.array(eps_vec, dtype=float)
        self.m = None
        self.v = None
        self.t = 0
    def step(self, theta, grad):
        self.t += 1
        if self.m is None:
            self.m = np.zeros_like(grad)
            self.v = np.zeros_like(grad)
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (grad ** 2)
        m_hat = self.m / (1.0 - self.beta1 ** self.t)
        v_hat = self.v / (1.0 - self.beta2 ** self.t)
        step_dir = m_hat / (np.sqrt(v_hat) + self.eps_vec)
        return theta - self.lr_vec * step_dir


def run_optimization(mesh, basis, tau_ref, W, theta_init, n_steps,
                     lambda_realiz=150.0, optimizer=None):
    d = len(basis)
    theta = theta_init.copy()
    I3 = np.eye(3)[None, None, :, :]
    history = {"loss": [], "viol": [], "theta": []}

    for step in range(n_steps):
        tau_dev = sum(theta[i] * basis[i] for i in range(d))
        tau = (2.0 / 3.0) * mesh.k[:, :, None, None] * I3 + tau_dev
        diff = tau - tau_ref
        loss_task = 0.5 * np.sum(W * (diff ** 2))
        grad_task = np.array([np.sum(W * diff * basis[i]) for i in range(d)])
        loss_realiz = 0.0
        grad_realiz = np.zeros(d)
        if lambda_realiz > 0:
            tau_sym = 0.5 * (tau + np.swapaxes(tau, 2, 3))
            eigvals, eigvecs = np.linalg.eigh(tau_sym)
            min_eig = eigvals[:, :, 0]
            min_vec = eigvecs[:, :, :, 0]
            mask = (min_eig < 0)
            if np.any(mask):
                viol = np.where(mask, min_eig, 0.0)
                loss_realiz = lambda_realiz * np.sum(mesh.W_grid * viol ** 2)
                for i in range(d):
                    vtv = np.einsum('abi,abij,abj->ab', min_vec, basis[i], min_vec)
                    grad_realiz[i] = 2.0 * lambda_realiz * np.sum(mesh.W_grid * viol * vtv)

        total_loss = loss_task + loss_realiz
        total_grad = grad_task + grad_realiz
        viol_frac = 1.0 - check_realizability(mesh, tau)
        history["loss"].append(float(total_loss))
        history["viol"].append(float(viol_frac))
        history["theta"].append(theta.tolist())

        theta = optimizer.step(theta, total_grad)

    return history, theta, optimizer


def main():
    mesh = ExtendedDuctMesh(h=1.0, Ny=48, Nz=48)
    W = mesh.W_grid[:, :, None, None]
    raw_basis = [mesh.T1, mesh.T2, mesh.T3, mesh.T4]
    norms = [float(np.sqrt(np.sum(W * T ** 2))) for T in raw_basis]
    norm_basis = [T / n for T, n in zip(raw_basis, norms)]
    lam = 150.0

    # Base Stratum 0 calibration
    opt0 = AdamOptimizer(lr=2e-3)
    _, theta_0_star, _ = run_optimization(
        mesh, [mesh.T1], mesh.tau_ref, W,
        theta_init=np.array([0.01]), n_steps=120,
        lambda_realiz=0.0, optimizer=opt0
    )
    th0_val = theta_0_star[0]
    print(f"Base Stratum 0 theta0* = {th0_val:.6f}")

    # =========================================================================
    # B2: MATCHED-LOSS B-RESTART (at step 18)
    # =========================================================================
    print("\n" + "="*70)
    print("B2: MATCHED-LOSS PATH B RESTART AT STEP 18")
    print("="*70)
    # Path B until step 18
    opt_b_pre = AdamOptimizer(lr=2e-3)
    h_b_pre, th_b_18, _ = run_optimization(
        mesh, raw_basis, mesh.tau_ref, W,
        theta_init=np.array([th0_val, 0.0, 0.0, 0.0]),
        n_steps=18, lambda_realiz=lam, optimizer=opt_b_pre
    )
    print(f"At step 18: loss = {h_b_pre['loss'][-1]:.6e}, viol = {h_b_pre['viol'][-1]*100:.2f}%")

    # Restart Adam at step 18 and continue for 202 steps (total 220)
    opt_b_post = AdamOptimizer(lr=2e-3)
    h_b_post, th_b_final_18, _ = run_optimization(
        mesh, raw_basis, mesh.tau_ref, W,
        theta_init=th_b_18.copy(), n_steps=202,
        lambda_realiz=lam, optimizer=opt_b_post
    )
    h_b_matched = {
        "loss": h_b_pre["loss"] + h_b_post["loss"],
        "viol": h_b_pre["viol"] + h_b_post["viol"],
        "theta": h_b_pre["theta"] + h_b_post["theta"]
    }
    peak_viol_post18 = max(h_b_post["viol"]) * 100
    overall_peak_viol_18 = max(h_b_matched["viol"]) * 100

    # Calculate rebound ratio post-restart
    post_losses = h_b_post["loss"]
    min_post = min(post_losses[:15])
    max_post = max(post_losses[:30])
    rebound_18 = max_post / min_post if min_post > 0 else 1.0

    print(f"Post-restart peak violation: {peak_viol_post18:.2f}%")
    print(f"Overall peak violation:      {overall_peak_viol_18:.2f}%")
    print(f"Post-restart rebound ratio:  {rebound_18:.3f}x")
    print(f"Final loss:                  {h_b_matched['loss'][-1]:.6e}")

    # =========================================================================
    # B6: PER-COORDINATE ADAM UNDER UNIT-NORMALIZED BASIS
    # =========================================================================
    print("\n" + "="*70)
    print("B6: PER-COORDINATE ADAM ON UNIT-NORMALIZED BASIS")
    print("="*70)
    alpha_base = 2e-3
    eps_val = 1e-16

    # Arm 1: Raw basis with Adam (eps = 1e-16)
    opt_raw_exact = AdamOptimizer(lr=alpha_base, eps=eps_val)
    h_raw_exact, th_raw_exact, _ = run_optimization(
        mesh, raw_basis, mesh.tau_ref, W,
        theta_init=np.array([th0_val, 0.0, 0.0, 0.0]),
        n_steps=220, lambda_realiz=lam, optimizer=opt_raw_exact
    )

    # Arm 2: Normalized basis with AdamPerCoord (alpha_n = alpha * ||T^(n)||, eps_n = eps / ||T^(n)||)
    lr_per_coord = [alpha_base * n for n in norms]
    eps_per_coord = [eps_val / n for n in norms]
    init_norm = np.array([th0_val * norms[0], 0.0, 0.0, 0.0])
    opt_norm_percoord = AdamPerCoord(lr_vec=lr_per_coord, eps_vec=eps_per_coord)
    h_norm_percoord, th_norm_percoord, _ = run_optimization(
        mesh, norm_basis, mesh.tau_ref, W,
        theta_init=init_norm, n_steps=220,
        lambda_realiz=lam, optimizer=opt_norm_percoord
    )

    # Convert normalized theta back to raw coordinates: theta_n = theta_norm_n / ||T^(n)||
    th_converted = np.array([th_norm_percoord[i] / norms[i] for i in range(4)])
    param_diff = np.linalg.norm(th_raw_exact - th_converted)
    max_loss_diff = max(abs(h_raw_exact["loss"][i] - h_norm_percoord["loss"][i]) for i in range(220))
    max_viol_diff = max(abs(h_raw_exact["viol"][i] - h_norm_percoord["viol"][i]) for i in range(220))
    rel_loss_diff = max(abs(h_raw_exact["loss"][i] - h_norm_percoord["loss"][i]) / h_raw_exact["loss"][i] for i in range(220))

    peak_viol_norm_percoord = max(h_norm_percoord["viol"]) * 100
    print(f"Raw basis exact peak viol:        {max(h_raw_exact['viol'])*100:.2f}%")
    print(f"Norm basis per-coord peak viol:   {peak_viol_norm_percoord:.2f}%")
    print(f"Max absolute loss discrepancy:    {max_loss_diff:.6e}")
    print(f"Max relative loss discrepancy:    {rel_loss_diff:.6e}")
    print(f"Parameter norm difference:        {param_diff:.6e}")
    print(f"Max violation discrepancy:        {max_viol_diff*100:.6e}%")

    # =========================================================================
    # B7: ADAM beta1 = 0 TO MATCHED FINAL LOSS (<= 1.3412e-9)
    # =========================================================================
    print("\n" + "="*70)
    print("B7: ADAM beta1 = 0 TO MATCHED LOSS (<= 1.3412e-9)")
    print("="*70)
    # Let's run beta1 = 0 with N = 400 steps
    opt_nobeta1_ext = AdamNoBeta1(lr=2e-3)
    h_nobeta1_ext, th_nobeta1_ext, _ = run_optimization(
        mesh, raw_basis, mesh.tau_ref, W,
        theta_init=np.array([th0_val, 0.0, 0.0, 0.0]),
        n_steps=450, lambda_realiz=lam, optimizer=opt_nobeta1_ext
    )
    step_matched_loss = None
    target_loss = 1.3412e-9
    for i, l_val in enumerate(h_nobeta1_ext["loss"]):
        if l_val <= target_loss:
            step_matched_loss = i
            break
    print(f"beta1=0 steps to reach L <= {target_loss:.4e}: {step_matched_loss}")
    if step_matched_loss is not None:
        peak_viol_matched = max(h_nobeta1_ext["viol"][:step_matched_loss+1]) * 100
        final_loss_matched = h_nobeta1_ext["loss"][step_matched_loss]
    else:
        peak_viol_matched = max(h_nobeta1_ext["viol"]) * 100
        final_loss_matched = h_nobeta1_ext["loss"][-1]
    overall_peak_viol_nobeta1 = max(h_nobeta1_ext["viol"]) * 100
    print(f"Peak viol up to matched loss: {peak_viol_matched:.2f}%")
    print(f"Overall peak viol over 450 steps: {overall_peak_viol_nobeta1:.2f}%")
    print(f"Final loss reached: {final_loss_matched:.6e}")

    # =========================================================================
    # B8: STEP-SIZE SWEEP FOR PLAIN GD & ADAM (Pareto plot)
    # =========================================================================
    print("\n" + "="*70)
    print("B8: STEP-SIZE SWEEP FOR PLAIN GD AND ADAM")
    print("="*70)
    target_tol = 1.4e-9

    # Sweep for Plain GD
    alphas_gd = np.logspace(0, 4, 15)  # from 1 to 10000
    results_sweep_gd = []
    for a in alphas_gd:
        opt_gd = PlainGD(lr=a)
        h, _, _ = run_optimization(
            mesh, raw_basis, mesh.tau_ref, W,
            theta_init=np.array([th0_val, 0.0, 0.0, 0.0]),
            n_steps=300, lambda_realiz=lam, optimizer=opt_gd
        )
        p_viol = max(h["viol"]) * 100
        steps_tol = None
        for step_idx, l_val in enumerate(h["loss"]):
            if l_val < target_tol:
                steps_tol = step_idx
                break
        final_l = h["loss"][-1]
        results_sweep_gd.append({
            "alpha": float(a),
            "peak_viol": float(p_viol),
            "steps_to_tol": steps_tol,
            "final_loss": float(final_l)
        })

    # Sweep for Adam
    alphas_adam = np.logspace(-4, -1.5, 12)  # from 1e-4 to 3e-2
    results_sweep_adam = []
    for a in alphas_adam:
        opt_adam = AdamOptimizer(lr=a)
        h, _, _ = run_optimization(
            mesh, raw_basis, mesh.tau_ref, W,
            theta_init=np.array([th0_val, 0.0, 0.0, 0.0]),
            n_steps=300, lambda_realiz=lam, optimizer=opt_adam
        )
        p_viol = max(h["viol"]) * 100
        steps_tol = None
        for step_idx, l_val in enumerate(h["loss"]):
            if l_val < target_tol:
                steps_tol = step_idx
                break
        final_l = h["loss"][-1]
        results_sweep_adam.append({
            "alpha": float(a),
            "peak_viol": float(p_viol),
            "steps_to_tol": steps_tol,
            "final_loss": float(final_l)
        })

    print("Plain GD sweep summary:")
    for r in results_sweep_gd:
        if r["steps_to_tol"] is not None:
            print(f"  alpha={r['alpha']:8.2f} | Peak viol={r['peak_viol']:5.2f}% | Steps to {target_tol}={r['steps_to_tol']}")

    print("\nAdam sweep summary:")
    for r in results_sweep_adam:
        if r["steps_to_tol"] is not None:
            print(f"  alpha={r['alpha']:8.5f} | Peak viol={r['peak_viol']:5.2f}% | Steps to {target_tol}={r['steps_to_tol']}")

    # Save Pareto figure
    fig, ax = plt.subplots(figsize=(7, 5))
    # Plot Plain GD
    conv_gd = [r for r in results_sweep_gd if r["steps_to_tol"] is not None]
    if conv_gd:
        ax.plot([r["steps_to_tol"] for r in conv_gd], [r["peak_viol"] for r in conv_gd],
                'o-', color='#542788', lw=1.8, ms=6, label='Plain GD (step-size sweep)')
        for r in conv_gd[::3]:
            ax.annotate(f"$\\alpha={r['alpha']:.0f}$", (r["steps_to_tol"], r["peak_viol"]),
                        fontsize=7.5, xytext=(5, 5), textcoords='offset points')

    # Plot Adam
    conv_adam = [r for r in results_sweep_adam if r["steps_to_tol"] is not None]
    if conv_adam:
        ax.plot([r["steps_to_tol"] for r in conv_adam], [r["peak_viol"] for r in conv_adam],
                's-', color='#E08214', lw=1.8, ms=6, label='Adam (learning-rate sweep)')
        for r in conv_adam[::3]:
            ax.annotate(f"$\\alpha={r['alpha']:.1e}$", (r["steps_to_tol"], r["peak_viol"]),
                        fontsize=7.5, xytext=(5, -10), textcoords='offset points')

    ax.set_xlabel(f'Steps to convergence ($L < {target_tol}$)', fontsize=10.5)
    ax.set_ylabel('Peak realizability violation (%)', fontsize=10.5)
    ax.set_title('Pareto Frontier: Convergence Speed vs Peak Realizability Violation', fontsize=11, fontweight='bold')
    ax.grid(True, ls=':', alpha=0.5)
    ax.legend(fontsize=9, loc='upper right')
    pareto_fig = PAPER_DIR / "figures" / "pareto_stepsize_sweep.png"
    plt.tight_layout()
    plt.savefig(pareto_fig, dpi=200)
    plt.close()
    print(f"Pareto plot saved to {pareto_fig}")

    # =========================================================================
    # SAVE INVESTIGATION DATA TO JSON
    # =========================================================================
    investigation_data = {
        "B2_matched_restart_step18": {
            "restart_step": 18,
            "loss_at_restart": float(h_b_pre["loss"][-1]),
            "post_restart_peak_viol": float(peak_viol_post18),
            "overall_peak_viol": float(overall_peak_viol_18),
            "rebound_ratio": float(rebound_18),
            "final_loss": float(h_b_matched["loss"][-1]),
            "history": {
                "loss": h_b_matched["loss"],
                "viol": h_b_matched["viol"]
            }
        },
        "B6_per_coordinate_adam": {
            "norms": norms,
            "alpha_per_coord": lr_per_coord,
            "peak_viol_raw": float(max(h_raw_exact["viol"]) * 100),
            "peak_viol_norm_percoord": float(peak_viol_norm_percoord),
            "param_diff": float(param_diff),
            "max_loss_diff": float(max_loss_diff),
            "rel_loss_diff": float(rel_loss_diff),
            "max_viol_diff": float(max_viol_diff)
        },
        "B7_beta1_zero_matched_loss": {
            "target_loss": target_loss,
            "steps_to_matched_loss": step_matched_loss,
            "peak_viol_at_matched_loss": float(peak_viol_matched),
            "overall_peak_viol": float(overall_peak_viol_nobeta1),
            "final_loss": float(final_loss_matched)
        },
        "B8_stepsize_sweep": {
            "plain_gd": results_sweep_gd,
            "adam": results_sweep_adam
        }
    }
    out_inv = DATA_DIR / "results_round3_investigations.json"
    with open(out_inv, "w") as f:
        json.dump(investigation_data, f, indent=2)
    print(f"\nAll investigation data saved to {out_inv}")


if __name__ == "__main__":
    main()
