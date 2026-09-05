#!/usr/bin/env python3
"""
verify_numbers.py

Verifies mathematical identities and consistency of reported numbers across:
1. paper/numbers.json
2. data/results_curvature_experiment.json
3. data/results_audit_experiments.json
4. data/results_round2_controls.json
5. data/results_round3_investigations.json
6. paper/paper.tex, README.md, paper/PAPER_DRAFT.md

Checks:
- Gram identities: G_ii = ||T^(i)||^2
- Condition ratio: (1 + |R_34|) / (1 - |R_34|) = 5.99
- Scale vs coupling decomposition: scale_disparity * kappa(G_tilde) ~ kappa(G) (30.8 * 2.08 = 64.1)
- Realizability floor: tau_ref violation == 1.56% (min eig = -5.10e-5)
- Path A hop delta_R == 0 exact
- Post-restart peaks: B-restart step 18 = 1.74%, B-restart step 100 = 1.22%
- Adam cold update transient: A-fresh deployed peak 5.90% vs A-carry 5.64%
- Scale invariance: per-coord Adam on normalized basis matches raw Adam to machine precision (< 1e-12)
"""

import sys
import json
import re
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"
PAPER_DIR = REPO_ROOT / "paper"

errors = []
warnings = []

def check(condition, message):
    if not condition:
        errors.append(f"[FAIL] {message}")
    else:
        print(f"[PASS] {message}")

# 1. Load data
try:
    with open(PAPER_DIR / "numbers.json") as f:
        nums = json.load(f)
    with open(DATA_DIR / "results_curvature_experiment.json") as f:
        orig = json.load(f)
    with open(DATA_DIR / "results_audit_experiments.json") as f:
        audit = json.load(f)
    with open(DATA_DIR / "results_round2_controls.json") as f:
        r2 = json.load(f)
    with open(DATA_DIR / "results_round3_investigations.json") as f:
        r3 = json.load(f)
    print("[PASS] Successfully loaded all JSON data files.")
except Exception as e:
    print(f"[FAIL] Error loading JSON files: {e}")
    sys.exit(1)

# 2. Gram Matrix Identities
gram = audit["gram_diagnostics"]
G = np.array(gram["G_full"])
norms = np.array(gram["L2_norms"])

for i in range(4):
    check(abs(G[i, i] - norms[i]**2) < 1e-12, f"Gram diagonal identity G_{i+1}{i+1} == ||T^({i+1})||^2")

r34 = gram["R_correlation"][2][3]
cond_r34 = (1.0 + abs(r34)) / (1.0 - abs(r34))
check(abs(cond_r34 - 5.99) < 0.02, f"Normalized 2x2 condition ratio: (1+|R_34|)/(1-|R_34|) = {cond_r34:.2f} == 5.99")

scale_disp = (norms[0] / norms[3])**2
kappa_norm = gram["kappa_unit_normalized"]
kappa_full = gram["kappa_full"]
coupling_factor = kappa_full / scale_disp
product = scale_disp * coupling_factor

check(abs(scale_disp - 30.82) < 0.05, f"Scale disparity (||T^(1)||/||T^(4)||)^2 = {scale_disp:.2f} == 30.8")
check(abs(kappa_norm - 5.99) < 0.02, f"Normalized Gram condition number kappa(G_tilde) = {kappa_norm:.2f} == 5.99")
check(abs(coupling_factor - 2.08) < 0.02, f"Cross-coupling inflation factor kappa(G)/scale_disp = {coupling_factor:.2f} == 2.08")
check(abs(kappa_full - 64.10) < 0.05, f"Full Gram condition number kappa(G) = {kappa_full:.1f} == 64.1")
check(abs(product - 64.10) < 0.05, f"Decomposition product 30.82 * 2.080 = {product:.1f} == 64.1")

# 3. Reference Field Realizability Floor
tau_viol = r2["reporting"]["tau_ref_violation"] * 100
min_eig = r2["reporting"]["tau_ref_min_eigenvalue"]
check(abs(tau_viol - 1.5625) < 0.01, f"tau_ref realizability violation = {tau_viol:.2f}% == 1.56%")
check(abs(min_eig - (-5.095e-5)) < 1e-7, f"tau_ref min eigenvalue = {min_eig:.2e} == -5.10e-5")

# 4. Path A Transport Realization Drift
delta_R = r2["reporting"]["delta_R_exact"]
check(delta_R == 0.0, f"Path A realization drift delta_R == 0.0 exact ({delta_R})")

# 5. Path A and B Key Trajectory Milestones
hA_loss = orig["history_A"]["loss"]
hA_viol = [v * 100 for v in orig["history_A"]["viol"]]
hB_loss = orig["history_B"]["loss"]
hB_viol = [v * 100 for v in orig["history_B"]["viol"]]

check(abs(hA_viol[99] - 5.64) < 0.01, f"Path A step 99 violation = {hA_viol[99]:.2f}% == 5.64%")
check(abs(hA_viol[100] - 5.64) < 0.01, f"Path A step 100 inherited violation = {hA_viol[100]:.2f}% == 5.64%")
check(abs(hA_viol[101] - 5.90) < 0.01, f"Path A step 101 deployed peak violation = {hA_viol[101]:.2f}% == 5.90%")
check(abs(max(hA_viol[:100]) - 6.08) < 0.01, f"Path A scaffold peak violation = {max(hA_viol[:100]):.2f}% == 6.08%")

check(abs(max(hB_viol) - 2.00) < 0.01, f"Path B peak violation = {max(hB_viol):.2f}% == 2.00%")
rebound_B = hB_loss[28] / hB_loss[21]
check(abs(rebound_B - 1.68) < 0.01, f"Path B rebound ratio = {rebound_B:.2f}x == 1.68x")

# 6. Decisive Controls
b_restart_100 = max(r2["control_1_B_restart"]["post_restart"]["viol"]) * 100
check(abs(b_restart_100 - 1.22) < 0.01, f"Path B-restart step 100 peak violation = {b_restart_100:.2f}% == 1.22%")

b_restart_18 = r3["B2_matched_restart_step18"]["post_restart_peak_viol"]
check(abs(b_restart_18 - 1.74) < 0.01, f"Path B-restart step 18 peak violation = {b_restart_18:.2f}% == 1.74%")
rebound_18 = r3["B2_matched_restart_step18"]["rebound_ratio"]
check(abs(rebound_18 - 3.12) < 0.02, f"Path B-restart step 18 rebound ratio = {rebound_18:.2f}x == 3.12x")

b2_loss_18 = nums["b2_loss_at_restart_step18"]["value"]
check(abs(b2_loss_18 - 4.343e-9) < 0.01e-9, f"B2 loss at restart step 18 = {b2_loss_18:.2e} == 4.34e-9 (< 4.73e-9)")
b2_loss_17 = nums["b2_loss_before_restart_step17"]["value"]
check(abs(b2_loss_17 - 5.839e-9) < 0.01e-9, f"B2 loss before restart step 17 = {b2_loss_17:.2e} == 5.84e-9 (> 4.73e-9)")

a_carry = max(r2["control_2_A_carry"]["deployed_carry"]["viol"]) * 100
check(abs(a_carry - 5.64) < 0.01, f"Path A-carry deployed max violation = {a_carry:.2f}% == 5.64%")
check(a_carry <= hA_viol[101], f"Path A-carry (5.64%) <= Path A-fresh reset (5.90%)")

conv_gd_A = max(r2["control_4_convergent_gd"]["path_A"]["viol"]) * 100
check(abs(conv_gd_A - 6.16) < 0.02, f"Convergent GD Path A peak violation = {conv_gd_A:.2f}% == 6.16%")

# 6b. Steps to Tolerance Strict Equality & Telemetry Verification
hC_loss = audit["experiments"]["cold_start"]["path_C"]["loss"]
check(nums["steps_to_tol_14_path_A"]["value"] == 123, f"Path A steps to tol < 1.4e-9 == 123 ({nums['steps_to_tol_14_path_A']['value']})")
check(nums["steps_to_tol_14_path_B"]["value"] == 59,  f"Path B steps to tol < 1.4e-9 == 59 ({nums['steps_to_tol_14_path_B']['value']})")
check(nums["steps_to_tol_14_path_C"]["value"] == 101, f"Path C steps to tol < 1.4e-9 == 101 ({nums['steps_to_tol_14_path_C']['value']})")

check(nums["steps_to_tol_1pct_path_A"]["value"] == 126, f"Path A steps to 1% final == 126 ({nums['steps_to_tol_1pct_path_A']['value']})")
check(nums["steps_to_tol_1pct_path_B"]["value"] == 89,  f"Path B steps to 1% final == 89 ({nums['steps_to_tol_1pct_path_B']['value']})")
check(nums["steps_to_tol_1pct_path_C"]["value"] == 111, f"Path C steps to 1% final == 111 ({nums['steps_to_tol_1pct_path_C']['value']})")

check(hA_loss[123] < 1.4e-9 and hA_loss[122] >= 1.4e-9, f"Telemetry check Path A step 123 (loss={hA_loss[123]:.3e} < 1.4e-9)")
check(hB_loss[59] < 1.4e-9 and hB_loss[58] >= 1.4e-9,   f"Telemetry check Path B step 59 (loss={hB_loss[59]:.3e} < 1.4e-9)")
check(hC_loss[101] < 1.4e-9 and hC_loss[100] >= 1.4e-9, f"Telemetry check Path C step 101 (loss={hC_loss[101]:.3e} < 1.4e-9)")

# 6c. Matched Convergence Rate Momentum Attribution
check(nums["matched_rate_steps"]["value"] == 76, f"Matched rate steps to tol == 76")
check(abs(nums["adam_matched_rate_peak_viol"]["value"] - 1.65) < 0.01, f"Full Adam matched-rate peak viol = {nums['adam_matched_rate_peak_viol']['value']:.2f}% == 1.65%")
check(abs(nums["beta1_zero_matched_rate_peak_viol"]["value"] - 1.30) < 0.01, f"Adam beta1=0 matched-rate peak viol = {nums['beta1_zero_matched_rate_peak_viol']['value']:.2f}% == 1.30%")
check(abs(nums["momentum_matched_rate_amplification"]["value"] - 1.27) < 0.02, f"Momentum amplification at matched rate = {nums['momentum_matched_rate_amplification']['value']:.2f}x == 1.27x")

# 7. Scale Invariance (B6)
rel_loss_diff = r3["B6_per_coordinate_adam"]["rel_loss_diff"]
param_diff = r3["B6_per_coordinate_adam"]["param_diff"]
check(rel_loss_diff < 1e-12, f"Scale invariance relative loss diff = {rel_loss_diff:.2e} < 1e-12")
check(param_diff < 1e-12, f"Scale invariance parameter diff = {param_diff:.2e} < 1e-12")

# 8. Convergence to Matched Parameter Vectors
th_A = np.array(r2["reporting"]["final_theta_A"])
th_B = np.array(r2["reporting"]["final_theta_B"])
th_C = np.array(r2["reporting"]["final_theta_C"])
dist_AB = np.linalg.norm(th_A - th_B)
dist_AC = np.linalg.norm(th_A - th_C)
check(dist_AB < 5e-5, f"||theta_A* - theta_B*|| = {dist_AB:.2e} < 5e-5")
check(dist_AC < 5e-5, f"||theta_A* - theta_C*|| = {dist_AC:.2e} < 5e-5")

# 9. Document Consistency Checks (paper.tex, README.md, PAPER_DRAFT.md)
print("\n--- Document Consistency Checks ---")
with open(PAPER_DIR / "paper.tex") as f:
    tex_content = f.read()
with open(REPO_ROOT / "README.md") as f:
    readme_content = f.read()
with open(PAPER_DIR / "PAPER_DRAFT.md") as f:
    draft_content = f.read()

# Check that banned phrases do not appear
banned = [
    "non-convex realizability boundary",
    "neither staged nor warm-started",
]
for phrase in banned:
    check(phrase not in tex_content, f"Banned phrase '{phrase}' absent from paper.tex")
    check(phrase not in readme_content, f"Banned phrase '{phrase}' absent from README.md")
    check(phrase not in draft_content, f"Banned phrase '{phrase}' absent from PAPER_DRAFT.md")

# Check that core verified numbers appear across all documents
core_values = [
    ("5.90%", "Path A deployed peak violation"),
    ("5.64%", "Path A inherited violation"),
    ("2.00%", "Path B peak violation"),
    ("1.22%", "Path C cold start peak violation"),
    ("1.74%", "Path B-restart step 18 post-restart peak"),
    ("1.30%", "Adam beta1=0 peak violation"),
    ("6.25%", "Plain GD peak violation"),
    ("1.56%", "tau_ref violation floor"),
    ("1.48%", "lambda=0 optimum residual"),
    ("1.04%", "lambda=150 baseline residual"),
    ("0.52%", "lambda=1500 stiff residual"),
    ("64.1",  "Gram condition number kappa(G_2)"),
    ("5.99",  "Normalized Gram condition number"),
    ("30.8",  "Scale disparity ratio"),
    ("2.08",  "Cross-coupling inflation factor"),
    ("1.68",  "Path B rebound ratio"),
    ("3.12",  "Path B step 18 restart rebound ratio"),
    ("1.65%", "Adam peak violation at matched convergence rate"),
    ("1.27",  "Momentum amplification at matched rate"),
    ("6.16%", "Convergent Plain GD Path A peak violation"),
    ("4.34",  "Path B restart loss at step 18"),
]

def contains_value(content, val):
    return val in content or val.replace("%", r"\%") in content

for val, desc in core_values:
    check(contains_value(tex_content, val), f"Value '{val}' ({desc}) present in paper.tex")
    check(contains_value(readme_content, val), f"Value '{val}' ({desc}) present in README.md")
    check(contains_value(draft_content, val), f"Value '{val}' ({desc}) present in PAPER_DRAFT.md")

# Check exact steps-to-tolerance consistency across documents
check("59" in tex_content and "59" in readme_content and "59" in draft_content, "Path B steps-to-tol '59' present in all 3 documents")
check("101" in tex_content and "101" in readme_content and "101" in draft_content, "Path C steps-to-tol '101' present in all 3 documents")
check("123" in tex_content and "123" in readme_content and "123" in draft_content, "Path A steps-to-tol '123' present in all 3 documents")

# 10. LaTeX Syntax, Reference & Citation Completeness
print("\n--- LaTeX Syntax, Reference, and Citation Integrity ---")
labels = set(m.group(1).strip() for m in re.finditer(r"\\label\{([^}]+)\}", tex_content))
refs = set(m.group(1).strip() for m in re.finditer(r"\\ref\{([^}]+)\}", tex_content))
missing_refs = refs - labels
check(len(missing_refs) == 0, f"All LaTeX references resolved (missing labels: {list(missing_refs)})")

cites = set()
for m in re.finditer(r"\\cite\{([^}]+)\}", tex_content):
    for key in m.group(1).split(","):
        cites.add(key.strip())

bib_keys = set()
for m in re.finditer(r"\\bibitem\{([^}]+)\}", tex_content):
    bib_keys.add(m.group(1).strip())

missing_bib = cites - bib_keys
check(len(missing_bib) == 0, f"All citations resolved in thebibliography (missing: {list(missing_bib)})")

# Check environment matching
lines = tex_content.splitlines()
env_stack = []
for i, line in enumerate(lines, 1):
    for m in re.finditer(r"\\(begin|end)\{([a-zA-Z*0-9]+)\}", line):
        action, env = m.groups()
        if action == "begin":
            env_stack.append((env, i))
        else:
            if env_stack:
                top_env, top_line = env_stack.pop()
                if top_env != env:
                    check(False, f"Mismatched LaTeX env \\end{{{env}}} at line {i} (expected \\end{{{top_env}}} from line {top_line})")
            else:
                check(False, f"Unmatched LaTeX env \\end{{{env}}} at line {i}")

check(len(env_stack) == 0, f"All LaTeX environments closed (unclosed: {len(env_stack)})")

# Check brace matching (handling comments and escaped braces)
open_braces = 0
for i, line in enumerate(lines, 1):
    s = line.replace(r"\%", "")
    idx = s.find("%")
    if idx >= 0:
        s = s[:idx]
    s = s.replace(r"\{", "").replace(r"\}", "")
    open_braces += s.count("{") - s.count("}")

check(open_braces == 0, f"All LaTeX braces balanced (net open: {open_braces})")

# Summary
print("\n" + "=" * 60)
if errors:
    print(f"VERIFICATION FAILED: {len(errors)} error(s) detected.")
    for err in errors:
        print(f"  {err}")
    sys.exit(1)
else:
    print("ALL NUMERICAL & SCHOLARSHIP VERIFICATIONS PASSED SUCCESSFULLY.")
    print("=" * 60)
    sys.exit(0)

