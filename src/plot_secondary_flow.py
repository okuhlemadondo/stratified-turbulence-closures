"""
plot_secondary_flow.py

Visualizes:
1. Turbulent Square Duct Streamwise Velocity U(y, z).
2. Secondary Velocity Vectors and Streamlines showing corner counter-rotating vortices.
3. Secondary Driving Stress (tau_yy - tau_zz) vs Linear Boussinesq prediction (identically zero).
"""

import os
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parent
PAPER_DIR = REPO_ROOT / "paper"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from prototype_square_duct import SquareDuctQuadrant

def plot_physics():
    mesh = SquareDuctQuadrant(h=1.0, Ny=64, Nz=64)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)
    
    # --------------------------------------------------------------------------
    # Panel 1: Streamwise Velocity U(y, z)
    # --------------------------------------------------------------------------
    ax1 = axes[0]
    cp1 = ax1.contourf(mesh.Z, mesh.Y, mesh.U, levels=30, cmap='viridis')
    cbar1 = plt.colorbar(cp1, ax=ax1)
    cbar1.set_label(r'$U / U_{\rm bulk}$', fontsize=11, fontweight='bold')
    ax1.set_title('(a) Streamwise Velocity $U(y, z)$', fontsize=12, fontweight='bold')
    ax1.set_xlabel('$z / h$ (Side wall at $z=0$)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('$y / h$ (Bottom wall at $y=0$)', fontsize=11, fontweight='bold')
    ax1.set_aspect('equal')
    
    # --------------------------------------------------------------------------
    # Panel 2: Secondary Flow Streamlines & Vectors (Prandtl 2nd Kind)
    # --------------------------------------------------------------------------
    ax2 = axes[1]
    # Interpolate onto a uniform grid for streamplot
    from scipy.interpolate import RegularGridInterpolator
    z_uni = np.linspace(mesh.z[0], mesh.z[-1], 100)
    y_uni = np.linspace(mesh.y[0], mesh.y[-1], 100)
    Z_uni, Y_uni = np.meshgrid(z_uni, y_uni)
    
    interp_W = RegularGridInterpolator((mesh.y, mesh.z), mesh.W, bounds_error=False, fill_value=0.0)
    interp_V = RegularGridInterpolator((mesh.y, mesh.z), mesh.V, bounds_error=False, fill_value=0.0)
    pts = np.stack([Y_uni, Z_uni], axis=-1)
    W_uni = interp_W(pts)
    V_uni = interp_V(pts)
    speed_uni = np.sqrt(W_uni**2 + V_uni**2)
    
    strm = ax2.streamplot(z_uni, y_uni, W_uni, V_uni, color=speed_uni, cmap='plasma', 
                          density=1.3, linewidth=1.5, arrowsize=1.2)
    cbar2 = plt.colorbar(strm.lines, ax=ax2)
    cbar2.set_label(r'$\sqrt{V^2 + W^2} / U_{\rm bulk}$', fontsize=11, fontweight='bold')
    ax2.set_title('(b) Secondary Flow: Corner Vortices', fontsize=12, fontweight='bold')
    ax2.set_xlabel('$z / h$', fontsize=11, fontweight='bold')
    ax2.set_ylabel('$y / h$', fontsize=11, fontweight='bold')
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Corner Bisector')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.set_aspect('equal')
    
    # --------------------------------------------------------------------------
    # Panel 3: Normal Stress Anisotropy (tau_yy - tau_zz) driving the vortices
    # --------------------------------------------------------------------------
    ax3 = axes[2]
    tau_aniso = mesh.tau_ref[:, :, 1, 1] - mesh.tau_ref[:, :, 2, 2]
    max_val = np.max(np.abs(tau_aniso))
    cp3 = ax3.contourf(mesh.Z, mesh.Y, tau_aniso, levels=30, cmap='RdBu_r', 
                       vmin=-max_val, vmax=max_val)
    cbar3 = plt.colorbar(cp3, ax=ax3)
    cbar3.set_label(r'$(\tau_{yy} - \tau_{zz})_{\rm ref}$', fontsize=11, fontweight='bold')
    ax3.set_title(r'(c) Anisotropic Driver: $(\tau_{yy} - \tau_{zz})$' + '\n' + r'[Boussinesq $\equiv 0$ everywhere]', 
                  fontsize=11, fontweight='bold')
    ax3.set_xlabel('$z / h$', fontsize=11, fontweight='bold')
    ax3.set_ylabel('$y / h$', fontsize=11, fontweight='bold')
    ax3.set_aspect('equal')
    
    plt.tight_layout()
    output_path = PAPER_DIR / "figures" / "square_duct_physics.png"
    plt.savefig(output_path)
    print(f"Physics figure saved to '{output_path}'.")

if __name__ == "__main__":
    plot_physics()
