# OpDiscovery-FluidMech: The Geometrization of Design Spaces

An implementation and validation framework for **The Geometrization of Design Spaces**, applied to invariant operator discovery and turbulence closure modeling in fluid mechanics.

## Overview

Configuration spaces in engineering and physics are hybrid:
- **Discrete**: Primitive selection, symbolic graph skeletons, differential operators.
- **Continuous**: Physical parameters, coefficients, scale lengths.
- **Semantic**: Conservation laws, Galilean and frame invariance, realizability boundaries (Lumley triangle).

This project formalizes the design space of turbulence closures as a **stratified design atlas** $(B, \{\Theta_t\}, \{R_t\}, L, E, \{T_e\}, \sim)$, where:
- Strata $\Theta_t$ are parameter manifolds of specific constitutive model families (e.g. Linear Boussinesq, Quadratic Pope expansion, Non-local gradient terms).
- Edits $e \in E$ are structural transitions between grammar skeletons.
- Transport maps $T_e: \Theta_t \to \Theta_{t'}$ carry calibrated parameters and optimizer state smoothly across structural mutations without catastrophic forgetting.
- Two-tiered controllability separates pointwise algebraic tensor completeness from operator-compositional PDE reachability.

## Canonical Benchmark: Turbulent Square Duct Flow

Turbulent square duct flow exhibits Prandtl's secondary motion of the second kind: 8 counter-rotating streamwise vortices driven by Reynolds stress anisotropy $(\tau_{yy} - \tau_{zz} \neq 0$ and $\tau_{yz} \neq 0$).
- Linear Boussinesq models ($S_{yy} - S_{zz} = 0$) mathematically fail with zero controllability alignment ($\cos \theta_{\text{sec}} \equiv 0$).
- Nonlinear Pope expansions span the required anisotropic tensor space ($\cos \theta_{\text{sec}} \gg 0$).
- Exact analytical transport $T_e(C_\mu^*) = [C_\mu^*, 0, 0]$ guarantees initial zero realization drift and inherited numerical stability.
