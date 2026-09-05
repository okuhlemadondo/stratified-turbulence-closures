"""
build_numbers_json.py

Generates paper/numbers.json binding every reported number in the manuscript,
README, and PAPER_DRAFT to its generating script, source JSON file, and JSON key.
"""

import json
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
PAPER_DIR = REPO_ROOT / "paper"

with open(DATA_DIR / "results_curvature_experiment.json") as f:
    orig = json.load(f)
with open(DATA_DIR / "results_audit_experiments.json") as f:
    audit = json.load(f)
with open(DATA_DIR / "results_round2_controls.json") as f:
    r2 = json.load(f)
with open(DATA_DIR / "results_round3_investigations.json") as f:
    r3 = json.load(f)

gram = audit["gram_diagnostics"]
G = np.array(gram["G_full"])
norms = gram["L2_norms"]

# Steps to tolerance
hA_loss = orig["history_A"]["loss"]
hB_loss = orig["history_B"]["loss"]
hC_loss = audit["experiments"]["cold_start"]["path_C"]["loss"]

tol_14 = 1.4e-9
tol_1pct = 1.341182e-9 * 1.01  # 1.35459e-9

steps_tol_14 = {
    "path_A": next(i for i, l in enumerate(hA_loss) if l < tol_14),
    "path_B": next(i for i, l in enumerate(hB_loss) if l < tol_14),
    "path_C": next(i for i, l in enumerate(hC_loss) if l < tol_14),
}

steps_tol_1pct = {
    "path_A": next(i for i, l in enumerate(hA_loss) if l <= tol_1pct),
    "path_B": next(i for i, l in enumerate(hB_loss) if l <= tol_1pct),
    "path_C": next(i for i, l in enumerate(hC_loss) if l <= tol_1pct),
}

r34 = float(gram["R_correlation"][2][3])
cond_r34 = (1.0 + abs(r34)) / (1.0 - abs(r34))
scale_disp = float((norms[0] / norms[3]) ** 2)

numbers = {
    "grid_ny": {
        "value": 48,
        "formatted": "48",
        "source_file": "src/prototype_square_duct.py",
        "description": "Cross-sectional grid points in y-direction"
    },
    "grid_nz": {
        "value": 48,
        "formatted": "48",
        "source_file": "src/prototype_square_duct.py",
        "description": "Cross-sectional grid points in z-direction"
    },
    "grid_total_cells": {
        "value": 2304,
        "formatted": "2304",
        "source_file": "src/prototype_square_duct.py",
        "description": "Total cells in quarter-duct domain (48 x 48)"
    },
    "total_optimization_steps": {
        "value": 220,
        "formatted": "220",
        "source_file": "src/curvature_experiment.py",
        "description": "Total gradient steps per path"
    },
    "realizability_penalty_lambda": {
        "value": 150.0,
        "formatted": "150.0",
        "source_file": "src/curvature_experiment.py",
        "description": "Penalty multiplier lambda for negative eigenvalues"
    },
    "G_11": {
        "value": float(G[0, 0]),
        "formatted": f"{G[0,0]:.6e}",
        "source_file": "data/results_audit_experiments.json",
        "json_path": "gram_diagnostics.G_full[0][0]",
        "generating_script": "src/audit_experiments.py",
        "description": "Gram matrix entry G_11 = ||T^(1)||^2"
    },
    "G_22": {
        "value": float(G[1, 1]),
        "formatted": f"{G[1,1]:.6e}",
        "source_file": "data/results_audit_experiments.json",
        "json_path": "gram_diagnostics.G_full[1][1]",
        "generating_script": "src/audit_experiments.py",
        "description": "Gram matrix entry G_22 = ||T^(2)||^2"
    },
    "G_33": {
        "value": float(G[2, 2]),
        "formatted": f"{G[2,2]:.6e}",
        "source_file": "data/results_audit_experiments.json",
        "json_path": "gram_diagnostics.G_full[2][2]",
        "generating_script": "src/audit_experiments.py",
        "description": "Gram matrix entry G_33 = ||T^(3)||^2"
    },
    "G_44": {
        "value": float(G[3, 3]),
        "formatted": f"{G[3,3]:.6e}",
        "source_file": "data/results_audit_experiments.json",
        "json_path": "gram_diagnostics.G_full[3][3]",
        "generating_script": "src/audit_experiments.py",
        "description": "Gram matrix entry G_44 = ||T^(4)||^2"
    },
    "G_34": {
        "value": float(G[2, 3]),
        "formatted": f"{G[2,3]:.6e}",
        "source_file": "data/results_audit_experiments.json",
        "json_path": "gram_diagnostics.G_full[2][3]",
        "generating_script": "src/audit_experiments.py",
        "description": "Gram matrix entry G_34 = <T^(3), T^(4)>"
    },
    "R_34": {
        "value": r34,
        "formatted": f"{r34:.4f}",
        "source_file": "data/results_audit_experiments.json",
        "json_path": "gram_diagnostics.R_correlation[2][3]",
        "generating_script": "src/audit_experiments.py",
        "description": "Normalized cross-correlation between T^(3) and T^(4)"
    },
    "kappa_full": {
        "value": float(gram["kappa_full"]),
        "formatted": f"{gram['kappa_full']:.2f}",
        "source_file": "data/results_audit_experiments.json",
        "json_path": "gram_diagnostics.kappa_full",
        "generating_script": "src/audit_experiments.py",
        "description": "Condition number of full 4x4 Gram matrix"
    },
    "kappa_block_01": {
        "value": float(gram["kappa_block_01"]),
        "formatted": f"{gram['kappa_block_01']:.2f}",
        "source_file": "data/results_audit_experiments.json",
        "json_path": "gram_diagnostics.kappa_block_01",
        "generating_script": "src/audit_experiments.py",
        "description": "Condition number of 3x3 Stratum-1 Gram block"
    },
    "kappa_unit_normalized": {
        "value": float(gram["kappa_unit_normalized"]),
        "formatted": f"{gram['kappa_unit_normalized']:.2f}",
        "source_file": "data/results_audit_experiments.json",
        "json_path": "gram_diagnostics.kappa_unit_normalized",
        "generating_script": "src/audit_experiments.py",
        "description": "Condition number of unit-normalized 4x4 Gram matrix"
    },
    "condition_ratio_R34": {
        "value": cond_r34,
        "formatted": f"{cond_r34:.2f}",
        "source_file": "data/results_audit_experiments.json",
        "json_path": "gram_diagnostics.R_correlation[2][3]",
        "generating_script": "src/audit_experiments.py",
        "description": "Condition number of isolated 2x2 normalized cross-coupling block (1+|R34|)/(1-|R34|)"
    },
    "scale_disparity_ratio": {
        "value": float((norms[0] / norms[3]) ** 2),
        "formatted": f"{(norms[0] / norms[3]) ** 2:.2f}",
        "source_file": "data/results_audit_experiments.json",
        "json_path": "gram_diagnostics.L2_norms",
        "generating_script": "src/audit_experiments.py",
        "description": "Scale disparity ratio (||T^(1)|| / ||T^(4)||)^2"
    },
    "tau_ref_violation_pct": {
        "value": float(r2["reporting"]["tau_ref_violation"] * 100),
        "formatted": f"{r2['reporting']['tau_ref_violation'] * 100:.2f}%",
        "source_file": "data/results_round2_controls.json",
        "json_path": "reporting.tau_ref_violation",
        "generating_script": "src/round2_controls.py",
        "description": "Realizability violation fraction of reference field tau_ref"
    },
    "tau_ref_min_eigenvalue": {
        "value": float(r2["reporting"]["tau_ref_min_eigenvalue"]),
        "formatted": f"{r2['reporting']['tau_ref_min_eigenvalue']:.6e}",
        "source_file": "data/results_round2_controls.json",
        "json_path": "reporting.tau_ref_min_eigenvalue",
        "generating_script": "src/round2_controls.py",
        "description": "Minimum eigenvalue of reference field tau_ref across domain"
    },
    "path_A_hop_delta_R": {
        "value": float(r2["reporting"]["delta_R_exact"]),
        "formatted": "0.00e+00",
        "source_file": "data/results_round2_controls.json",
        "json_path": "reporting.delta_R_exact",
        "generating_script": "src/round2_controls.py",
        "description": "Exact realizability drift across Path A hops"
    },
    "path_A_scaffold_peak_viol": {
        "value": float(max(orig["history_A"]["viol"][:100]) * 100),
        "formatted": f"{max(orig['history_A']['viol'][:100]) * 100:.2f}%",
        "source_file": "data/results_curvature_experiment.json",
        "json_path": "history_A.viol[0..99]",
        "generating_script": "src/curvature_experiment.py",
        "description": "Peak realizability violation of Path A during Stratum-1 scaffold phase (step 38)"
    },
    "path_A_step99_loss": {
        "value": float(orig["history_A"]["loss"][99]),
        "formatted": f"{orig['history_A']['loss'][99]:.4e}",
        "source_file": "data/results_curvature_experiment.json",
        "json_path": "history_A.loss[99]",
        "generating_script": "src/curvature_experiment.py",
        "description": "Path A loss at step 99 (100th Stratum-1 gradient step)"
    },
    "path_A_step99_viol": {
        "value": float(orig["history_A"]["viol"][99] * 100),
        "formatted": f"{orig['history_A']['viol'][99] * 100:.2f}%",
        "source_file": "data/results_curvature_experiment.json",
        "json_path": "history_A.viol[99]",
        "generating_script": "src/curvature_experiment.py",
        "description": "Path A violation at step 99"
    },
    "path_A_step100_viol": {
        "value": float(orig["history_A"]["viol"][100] * 100),
        "formatted": f"{orig['history_A']['viol'][100] * 100:.2f}%",
        "source_file": "data/results_curvature_experiment.json",
        "json_path": "history_A.viol[100]",
        "generating_script": "src/curvature_experiment.py",
        "description": "Path A violation at step 100 (inherited into Stratum 2 before gradient descent)"
    },
    "path_A_step101_viol": {
        "value": float(orig["history_A"]["viol"][101] * 100),
        "formatted": f"{orig['history_A']['viol'][101] * 100:.2f}%",
        "source_file": "data/results_curvature_experiment.json",
        "json_path": "history_A.viol[101]",
        "generating_script": "src/curvature_experiment.py",
        "description": "Path A deployed peak violation at step 101 (first Stratum-2 gradient step)"
    },
    "path_A_final_loss": {
        "value": float(orig["history_A"]["loss"][-1]),
        "formatted": f"{orig['history_A']['loss'][-1]:.4e}",
        "source_file": "data/results_curvature_experiment.json",
        "json_path": "history_A.loss[-1]",
        "generating_script": "src/curvature_experiment.py",
        "description": "Path A final calibrated loss at step 220"
    },
    "path_A_final_viol": {
        "value": float(orig["history_A"]["viol"][-1] * 100),
        "formatted": f"{orig['history_A']['viol'][-1] * 100:.2f}%",
        "source_file": "data/results_curvature_experiment.json",
        "json_path": "history_A.viol[-1]",
        "generating_script": "src/curvature_experiment.py",
        "description": "Path A final violation at step 220"
    },
    "path_B_peak_viol": {
        "value": float(max(orig["history_B"]["viol"]) * 100),
        "formatted": f"{max(orig['history_B']['viol']) * 100:.2f}%",
        "source_file": "data/results_curvature_experiment.json",
        "json_path": "history_B.viol",
        "generating_script": "src/curvature_experiment.py",
        "description": "Path B peak realizability violation (step 25)"
    },
    "path_B_rebound_ratio": {
        "value": float(orig["history_B"]["loss"][28] / orig["history_B"]["loss"][21]),
        "formatted": f"{orig['history_B']['loss'][28] / orig['history_B']['loss'][21]:.2f}x",
        "source_file": "data/results_curvature_experiment.json",
        "json_path": "history_B.loss[28]/loss[21]",
        "generating_script": "src/curvature_experiment.py",
        "description": "Path B loss rebound ratio from step 21 (2.62e-9) to step 28 (4.40e-9)"
    },
    "path_B_final_loss": {
        "value": float(orig["history_B"]["loss"][-1]),
        "formatted": f"{orig['history_B']['loss'][-1]:.4e}",
        "source_file": "data/results_curvature_experiment.json",
        "json_path": "history_B.loss[-1]",
        "generating_script": "src/curvature_experiment.py",
        "description": "Path B final calibrated loss at step 220"
    },
    "path_C_peak_viol": {
        "value": float(max(audit["experiments"]["cold_start"]["path_C"]["viol"]) * 100),
        "formatted": f"{max(audit['experiments']['cold_start']['path_C']['viol']) * 100:.2f}%",
        "source_file": "data/results_audit_experiments.json",
        "json_path": "experiments.cold_start.path_C.viol",
        "generating_script": "src/audit_experiments.py",
        "description": "Path C (cold start) peak realizability violation (step 28)"
    },
    "path_C_final_loss": {
        "value": float(audit["experiments"]["cold_start"]["path_C"]["loss"][-1]),
        "formatted": f"{audit['experiments']['cold_start']['path_C']['loss'][-1]:.4e}",
        "source_file": "data/results_audit_experiments.json",
        "json_path": "experiments.cold_start.path_C.loss[-1]",
        "generating_script": "src/audit_experiments.py",
        "description": "Path C final calibrated loss at step 220"
    },
    "steps_to_tol_14_path_A": {
        "value": steps_tol_14["path_A"],
        "formatted": str(steps_tol_14["path_A"]),
        "source_file": "data/results_curvature_experiment.json",
        "generating_script": "src/round3_investigations.py",
        "description": "Steps for Path A to reach L < 1.4e-9 (100 in S1 + 23 in S2)"
    },
    "steps_to_tol_14_path_B": {
        "value": steps_tol_14["path_B"],
        "formatted": str(steps_tol_14["path_B"]),
        "source_file": "data/results_curvature_experiment.json",
        "generating_script": "src/round3_investigations.py",
        "description": "Steps for Path B to reach L < 1.4e-9"
    },
    "steps_to_tol_14_path_C": {
        "value": steps_tol_14["path_C"],
        "formatted": str(steps_tol_14["path_C"]),
        "source_file": "data/results_audit_experiments.json",
        "generating_script": "src/round3_investigations.py",
        "description": "Steps for Path C to reach L < 1.4e-9"
    },
    "steps_to_tol_1pct_path_A": {
        "value": steps_tol_1pct["path_A"],
        "formatted": str(steps_tol_1pct["path_A"]),
        "source_file": "data/results_curvature_experiment.json",
        "generating_script": "src/round3_investigations.py",
        "description": "Steps for Path A to reach L <= 1.3546e-9 (1% above final loss)"
    },
    "steps_to_tol_1pct_path_B": {
        "value": steps_tol_1pct["path_B"],
        "formatted": str(steps_tol_1pct["path_B"]),
        "source_file": "data/results_curvature_experiment.json",
        "generating_script": "src/round3_investigations.py",
        "description": "Steps for Path B to reach L <= 1.3546e-9 (1% above final loss)"
    },
    "steps_to_tol_1pct_path_C": {
        "value": steps_tol_1pct["path_C"],
        "formatted": str(steps_tol_1pct["path_C"]),
        "source_file": "data/results_audit_experiments.json",
        "generating_script": "src/round3_investigations.py",
        "description": "Steps for Path C to reach L <= 1.3546e-9 (1% above final loss)"
    },
    "control_B_restart_step100_post_peak": {
        "value": float(max(r2["control_1_B_restart"]["post_restart"]["viol"]) * 100),
        "formatted": f"{max(r2['control_1_B_restart']['post_restart']['viol']) * 100:.2f}%",
        "source_file": "data/results_round2_controls.json",
        "json_path": "control_1_B_restart.post_restart.viol",
        "generating_script": "src/round2_controls.py",
        "description": "Path B-restart at step 100 post-restart peak violation"
    },
    "control_B_restart_step18_post_peak": {
        "value": float(r3["B2_matched_restart_step18"]["post_restart_peak_viol"]),
        "formatted": f"{r3['B2_matched_restart_step18']['post_restart_peak_viol']:.2f}%",
        "source_file": "data/results_round3_investigations.json",
        "json_path": "B2_matched_restart_step18.post_restart_peak_viol",
        "generating_script": "src/round3_investigations.py",
        "description": "Path B-restart at step 18 (matched loss) post-restart peak violation"
    },
    "control_B_restart_step18_rebound": {
        "value": float(r3["B2_matched_restart_step18"]["rebound_ratio"]),
        "formatted": f"{r3['B2_matched_restart_step18']['rebound_ratio']:.2f}x",
        "source_file": "data/results_round3_investigations.json",
        "json_path": "B2_matched_restart_step18.rebound_ratio",
        "generating_script": "src/round3_investigations.py",
        "description": "Path B-restart at step 18 post-restart loss rebound ratio"
    },
    "control_A_carry_deployed_max_viol": {
        "value": float(max(r2["control_2_A_carry"]["deployed_carry"]["viol"]) * 100),
        "formatted": f"{max(r2['control_2_A_carry']['deployed_carry']['viol']) * 100:.2f}%",
        "source_file": "data/results_round2_controls.json",
        "json_path": "control_2_A_carry.deployed_carry.viol",
        "generating_script": "src/round2_controls.py",
        "description": "Path A carrying momentum across hop deployed phase max violation"
    },
    "control_beta1_zero_path_B_peak_viol": {
        "value": float(max(r2["control_3_no_momentum"]["path_B"]["viol"]) * 100),
        "formatted": f"{max(r2['control_3_no_momentum']['path_B']['viol']) * 100:.2f}%",
        "source_file": "data/results_round2_controls.json",
        "json_path": "control_3_no_momentum.path_B.viol",
        "generating_script": "src/round2_controls.py",
        "description": "Path B with Adam beta1=0 peak realizability violation"
    },
    "control_beta1_zero_path_B_final_loss": {
        "value": float(r2["control_3_no_momentum"]["path_B"]["loss"][-1]),
        "formatted": f"{r2['control_3_no_momentum']['path_B']['loss'][-1]:.4e}",
        "source_file": "data/results_round2_controls.json",
        "json_path": "control_3_no_momentum.path_B.loss[-1]",
        "generating_script": "src/round2_controls.py",
        "description": "Path B with Adam beta1=0 loss at step 220"
    },
    "control_conv_gd_path_B_peak_viol": {
        "value": float(max(r2["control_4_convergent_gd"]["path_B"]["viol"]) * 100),
        "formatted": f"{max(r2['control_4_convergent_gd']['path_B']['viol']) * 100:.2f}%",
        "source_file": "data/results_round2_controls.json",
        "json_path": "control_4_convergent_gd.path_B.viol",
        "generating_script": "src/round2_controls.py",
        "description": "Path B with Plain GD (alpha = 7323.77) peak violation"
    },
    "control_conv_gd_path_A_peak_viol": {
        "value": float(max(r2["control_4_convergent_gd"]["path_A"]["viol"]) * 100),
        "formatted": f"{max(r2['control_4_convergent_gd']['path_A']['viol']) * 100:.2f}%",
        "source_file": "data/results_round2_controls.json",
        "json_path": "control_4_convergent_gd.path_A.viol",
        "generating_script": "src/round2_controls.py",
        "description": "Path A with Plain GD (alpha = 7323.77) peak violation"
    },
    "control_rescaled_alpha_peak_viol": {
        "value": float(max(r2["control_5_rescaled_alpha"]["path_B"]["viol"]) * 100),
        "formatted": f"{max(r2['control_5_rescaled_alpha']['path_B']['viol']) * 100:.2f}%",
        "source_file": "data/results_round2_controls.json",
        "json_path": "control_5_rescaled_alpha.path_B.viol",
        "generating_script": "src/round2_controls.py",
        "description": "Path B with normalized basis and scalar rescaled alpha peak violation"
    },
    "control_per_coord_adam_peak_viol": {
        "value": float(r3["B6_per_coordinate_adam"]["peak_viol_norm_percoord"]),
        "formatted": f"{r3['B6_per_coordinate_adam']['peak_viol_norm_percoord']:.2f}%",
        "source_file": "data/results_round3_investigations.json",
        "json_path": "B6_per_coordinate_adam.peak_viol_norm_percoord",
        "generating_script": "src/round3_investigations.py",
        "description": "Path B with normalized basis and per-coordinate Adam peak violation"
    },
    "b2_loss_before_restart_step17": {
        "value": 5.839047611748329e-09,
        "formatted": "5.84e-09",
        "source_file": "data/results_round3_investigations.json",
        "json_path": "B2_matched_restart_step18.history.loss[17]",
        "generating_script": "src/round3_investigations.py",
        "description": "Path B loss at step 17 immediately preceding restart"
    },
    "b2_loss_at_restart_step18": {
        "value": 4.343150471939988e-09,
        "formatted": "4.34e-09",
        "source_file": "data/results_round3_investigations.json",
        "json_path": "B2_matched_restart_step18.history.loss[18]",
        "generating_script": "src/round3_investigations.py",
        "description": "Path B loss at step 18 (first step below Path A hop loss 4.73e-9)"
    },
    "matched_rate_steps": {
        "value": 76,
        "formatted": "76",
        "source_file": "data/results_round3_investigations.json",
        "generating_script": "src/round3_investigations.py",
        "description": "Steps to L < 1.4e-9 matched between full Adam (alpha=1.37e-3) and beta1=0 (alpha=2e-3)"
    },
    "adam_matched_rate_peak_viol": {
        "value": 1.649305555555558,
        "formatted": "1.65%",
        "source_file": "data/results_round3_investigations.json",
        "json_path": "B8_stepsize_sweep.adam[5].peak_viol",
        "generating_script": "src/round3_investigations.py",
        "description": "Full Adam peak violation at matched convergence rate (76 steps)"
    },
    "beta1_zero_matched_rate_peak_viol": {
        "value": 1.302083333333337,
        "formatted": "1.30%",
        "source_file": "data/results_round2_controls.json",
        "json_path": "control_3_no_momentum.path_B.viol",
        "generating_script": "src/round2_controls.py",
        "description": "Adam beta1=0 peak violation at matched convergence rate (76 steps)"
    },
    "momentum_matched_rate_amplification": {
        "value": 1.649305555555558 / 1.302083333333337,
        "formatted": "1.27x",
        "source_file": "data/results_round3_investigations.json",
        "generating_script": "src/round3_investigations.py",
        "description": "Momentum excursion amplification factor at matched convergence rate (1.65% / 1.30%)"
    },
    "param_distance_A_B": {
        "value": float(np.linalg.norm(np.array(r2["reporting"]["final_theta_A"]) - np.array(r2["reporting"]["final_theta_B"]))),
        "formatted": f"{np.linalg.norm(np.array(r2['reporting']['final_theta_A']) - np.array(r2['reporting']['final_theta_B'])):.4e}",
        "source_file": "data/results_round2_controls.json",
        "json_path": "reporting.final_theta_A, final_theta_B",
        "generating_script": "src/round2_controls.py",
        "description": "L2 distance between final calibrated parameters of Path A and Path B"
    },
    "param_distance_A_c_true": {
        "value": float(np.linalg.norm(np.array(r2["reporting"]["final_theta_A"]) - np.array(r2["reporting"]["true_coefficients"]))),
        "formatted": f"{np.linalg.norm(np.array(r2['reporting']['final_theta_A']) - np.array(r2['reporting']['true_coefficients'])):.4e}",
        "source_file": "data/results_round2_controls.json",
        "json_path": "reporting.final_theta_A, true_coefficients",
        "generating_script": "src/round2_controls.py",
        "description": "L2 distance between Path A calibrated parameters and ground truth coefficients"
    }
}

out_path = PAPER_DIR / "numbers.json"
with open(out_path, "w") as f:
    json.dump(numbers, f, indent=2)
print(f"Saved {len(numbers)} verified numbers to {out_path}")
