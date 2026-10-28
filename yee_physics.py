"""Minimal 3D Yee-FDTD physics, with no visualization."""

import numpy as np
import cupy as cp
import math
import sources
from sources import TemporalBehaviours

eps_fixed = 8.854*(10**(-12))  # Vacuum permittivity (F/m)
mu_fixed = 1.257*(10**(-6))  # Vacuum permeability (H/m)
c = 1.0 / math.sqrt(eps_fixed * mu_fixed)
N = 100
dx = dy = dz = 0.5
spacing = (dx, dy, dz)

# Each field is one array: [component, x, y, z].
E = cp.zeros((3, N, N, N))  # Ex, Ey, Ez
B = cp.zeros((3, N, N, N))  # Bx, By, Bz
eps= cp.full_like(E, eps_fixed)
mu = cp.full_like(B, mu_fixed) 

def compute_stable_dt(safety=0.85):
    """Conservative CFL step for positive, linear isotropic epsilon/mu maps."""
    global eps_fixed, mu_fixed, dx, dy, dz
    if not (0 < safety < 1):
        raise ValueError("safety must be between 0 and 1")
    eps_min = float(cp.min(eps).item())
    mu_min = float(cp.min(mu).item())
    if eps_min <= 0 or mu_min <= 0:
        raise ValueError("This basic FDTD solver requires epsilon > 0 and mu > 0")
    vmax = 1.0 / math.sqrt(eps_min * mu_min)
    return safety / (vmax * math.sqrt(dx**-2 + dy**-2 + dz**-2))

dt = compute_stable_dt()

'''def forward(f, axis):
    """Forward difference with periodic boundaries."""
    pad_width = [(0, 0)] * f.ndim
    pad_width[axis] = (0, 1)
    f_padded = cp.pad(f, pad_width, mode='constant')
    hi = [slice(None)] * f.ndim
    hi[axis] = slice(1, None)
    return (f_padded[tuple(hi)] - f) / spacing[axis]


def backward(f, axis):  
    """Backward difference with periodic boundaries."""
    pad_width = [(0, 0)] * f.ndim
    pad_width[axis] = (1, 0)
    f_padded = cp.pad(f, pad_width, mode='constant')
    lo = [slice(None)] * f.ndim
    lo[axis] = slice(0, -1)
    return (f - f_padded[tuple(lo)]) / spacing[axis]'''


# Fast forward difference with zero-valued boundaries.
_forward_kernel = cp.ElementwiseKernel(
    'raw T f, int32 side, int32 stride, float64 inv_spacing',
    'T result',
    '''
    int coord = (i / stride) % side;

    T next_value = (coord == side - 1)
        ? (T)0
        : f[i + stride];

    result = (next_value - f[i]) * inv_spacing;
    ''',
    'forward_zero_boundary'
)

# Fast backward difference with zero-valued boundaries.
_backward_kernel = cp.ElementwiseKernel(
    'raw T f, int32 side, int32 stride, float64 inv_spacing',
    'T result',
    '''
    int coord = (i / stride) % side;

    T previous_value = (coord == 0)
        ? (T)0
        : f[i - stride];

    result = (f[i] - previous_value) * inv_spacing;
    ''',
    'backward_zero_boundary'
)

def forward(f, axis):
    """Forward difference without allocating padded arrays."""
    out = cp.empty_like(f)
    _forward_kernel(
        f,
        f.shape[axis],
        f.strides[axis] // f.itemsize,
        1.0 / spacing[axis],
        out
    )
    return out


def backward(f, axis):
    """Backward difference without allocating padded arrays."""
    out = cp.empty_like(f)
    _backward_kernel(
        f,
        f.shape[axis],
        f.strides[axis] // f.itemsize,
        1.0 / spacing[axis],
        out
    )
    return out

test = cp.arange(125, dtype=cp.float64).reshape(5, 5, 5)

for axis in range(3):
    pad_f = [(0, 0)] * 3
    pad_f[axis] = (0, 1)

    old_forward = cp.diff(
        cp.pad(test, pad_f, mode="constant"),
        axis=axis
    ) / spacing[axis]

    pad_b = [(0, 0)] * 3
    pad_b[axis] = (1, 0)

    old_backward = cp.diff(
        cp.pad(test, pad_b, mode="constant"),
        axis=axis
    ) / spacing[axis]

    print(
        f"Axis {axis}:",
        bool(cp.allclose(forward(test, axis), old_forward).item()),
        bool(cp.allclose(backward(test, axis), old_backward).item())
    )

def curl(F, difference):
    """Calculate the three components of curl(F)."""
    Fx, Fy, Fz = F
    return cp.stack([
        difference(Fz, 1) - difference(Fy, 2),
        difference(Fx, 2) - difference(Fz, 0),
        difference(Fy, 0) - difference(Fx, 1),
    ])


def divergence(F, difference):
    """Calculate the discrete divergence of F."""
    return sum(difference(F[a], a) for a in range(3))


current_sources = (
    {
        "source shape": sources.SourceShapes.dipole(
            axis = 1,
            room_length = N,
            source_position = (30, 50, 50)
        ),
        "temporal behaviour": lambda t: sources.TemporalBehaviours.packet_burst(t)
    },
    {
        "source shape": sources.SourceShapes.dipole(
            axis = 2,
            room_length = N,
            source_position = (70, 50, 50),
        ),
        "temporal behaviour": lambda t: sources.TemporalBehaviours.packet_burst(t)
    }
)


J_shape_cpu = np.zeros((3, N, N, N))
for current_source in current_sources: 
    J_shape_cpu += current_source["source shape"]
J_shape = cp.asarray(J_shape_cpu) # Move to GPU ONCE
del J_shape_cpu


def step(n):
    """Advance the fields by one time step."""
    global E, B, H

    t = (n + 0.5) * dt
    tau = t*c/dx

    pulse= TemporalBehaviours.packet_burst(tau)
    #pulse = np.sin(2 * np.pi * 0.08 * tau) * np.exp(-((tau - 9.0) / 3.5)**2) #to revert change, replace tau with t
    J = eps_fixed*c*pulse * J_shape  #to revert change remove eps_fixed and c

    # Faraday's law: update B using the current E.
    B -= dt * curl(E, forward)
    H = B / mu
    # Ampere-Maxwell law: update E using the new B.
    E += dt * (curl(H, backward) - J) / eps


def check_constraints():
    """Return the divergence diagnostics used by the original script."""
    #div_e = np.max(np.abs(divergence(E, backward)))
    div_e = cp.max(cp.abs(divergence(eps*E, backward)))
    div_b = cp.max(cp.abs(divergence(B, forward)))
    div_j = cp.max(cp.abs(divergence(J_shape, backward)))
    return(
        float(div_e.item()),
        float(div_b.item()), 
        float(div_j.item())
    )


if __name__ == "__main__":
    # Run the physics by itself, without opening a window.
    for n in range(50):
        step(n)

    div_e, div_b, div_j = check_constraints()
    print(f"max |div E|: {div_e:.3e}")
    print(f"max |div B|: {div_b:.3e}")
    print(f"max |div J|: {div_j:.3e}")
