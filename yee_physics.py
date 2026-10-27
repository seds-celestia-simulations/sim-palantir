"""Minimal 3D Yee-FDTD physics, with no visualization."""

import numpy as np

eps_fixed = 8.854*(10**(-12))  # Vacuum permittivity (F/m)
mu_fixed = 1.257*(10**(-6))  # Vacuum permeability (H/m)
c = 1.0 / np.sqrt(eps_fixed * mu_fixed)
N = 100
dx = dy = dz = 0.5
spacing = (dx, dy, dz)

# Each field is one array: [component, x, y, z].
E = np.zeros((3, N, N, N))  # Ex, Ey, Ez
B = np.zeros((3, N, N, N))  # Bx, By, Bz
eps= np.full_like(E, eps_fixed)
mu = np.full_like(B, mu_fixed) 

def compute_stable_dt(safety=0.85):
    """Conservative CFL step for positive, linear isotropic epsilon/mu maps."""
    global eps_fixed, mu_fixed, dx, dy, dz
    if not (0 < safety < 1):
        raise ValueError("safety must be between 0 and 1")
    eps_min = float(np.min(eps))
    mu_min = float(np.min(mu))
    if eps_min <= 0 or mu_min <= 0:
        raise ValueError("This basic FDTD solver requires epsilon > 0 and mu > 0")
    vmax = 1.0 / np.sqrt(eps_min * mu_min)
    return safety / (vmax * np.sqrt(dx**-2 + dy**-2 + dz**-2))

dt = compute_stable_dt()

def forward(f, axis):
    """Forward difference with periodic boundaries."""
    return (np.roll(f, -1, axis=axis) - f) / spacing[axis]


def backward(f, axis):
    """Backward difference with periodic boundaries."""
    return (f - np.roll(f, 1, axis=axis)) / spacing[axis]


def curl(F, difference):
    """Calculate the three components of curl(F)."""
    Fx, Fy, Fz = F
    return np.stack([
        difference(Fz, 1) - difference(Fy, 2),
        difference(Fx, 2) - difference(Fz, 0),
        difference(Fy, 0) - difference(Fx, 1),
    ])


def divergence(F, difference):
    """Calculate the discrete divergence of F."""
    return sum(difference(F[a], a) for a in range(3))


# Build a localized, divergence-free current source.
x, y, z = np.ogrid[:N, :N, :N]
r2 = (x - N / 2)**2 + (y - N / 2)**2 + (z - N / 2)**2

A = np.zeros_like(E)
A[2] = np.exp(-r2 / (2 * 2.0**2))
J_shape = curl(A, backward)


def step(n):
    """Advance the fields by one time step."""
    global E, B, H

    # Short sinusoidal pulse.
    t = (n + 0.5) * dt
    tau = t*c/dx

    pulse = np.sin(2 * np.pi * 0.08 * tau) * np.exp(-((tau - 9.0) / 3.5)**2) #to revert change, replace tau with t
    J = eps_fixed*c*pulse * J_shape  #to revert change remove eps_fixed and c

    # Faraday's law: update B using the current E.
    B -= dt * curl(E, forward)
    H = B / mu
    # Ampere-Maxwell law: update E using the new B.
    E += dt * (curl(H, backward) - J) / eps


def check_constraints():
    """Return the divergence diagnostics used by the original script."""
    #div_e = np.max(np.abs(divergence(E, backward)))
    div_e = np.max(np.abs(divergence(eps*E, backward)))
    div_b = np.max(np.abs(divergence(B, forward)))
    div_j = np.max(np.abs(divergence(J_shape, backward)))
    return div_e, div_b, div_j


if __name__ == "__main__":
    # Run the physics by itself, without opening a window.
    for n in range(50):
        step(n)

    div_e, div_b, div_j = check_constraints()
    print(f"max |div E|: {div_e:.3e}")
    print(f"max |div B|: {div_b:.3e}")
    print(f"max |div J|: {div_j:.3e}")
