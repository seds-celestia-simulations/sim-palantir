"""Minimal 3D Yee-FDTD physics, with no visualization."""


import cupy as cp

from math import sqrt

import sources


eps_fixed = 8.854 * 10e-12  # Vacuum permittivity (F/m)
mu_fixed = 1.257 * 10e-6  # Vacuum permeability (H/m)

N = 100
dx = dy = dz = 0.5
spacing = (dx, dy, dz)

E = cp.zeros((3, N, N, N))  # Ex, Ey, Ez
B = cp.zeros((3, N, N, N))  # Bx, By, Bz
eps = cp.full_like(E, eps_fixed)
mu = cp.full_like(B, mu_fixed) 
c = 1.0 / sqrt(eps_fixed * mu_fixed)


def compute_stable_dt(safety=0.85):
    """Conservative CFL step for positive, linear isotropic epsilon/mu maps."""
    global eps_fixed, mu_fixed, dx, dy, dz
    if not (0 < safety < 1):
        raise ValueError("safety must be between 0 and 1")
    eps_min = float(cp.min(eps).item())
    mu_min = float(cp.min(mu).item())
    if eps_min <= 0 or mu_min <= 0:
        raise ValueError("This basic FDTD solver requires epsilon > 0 and mu > 0")
    vmax = 1.0 / sqrt(eps_min * mu_min)
    return safety / (vmax * sqrt(dx**-2 + dy**-2 + dz**-2))

dt = compute_stable_dt()

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

# Dynamic Sources Collection
current_sources = []
J_shape = cp.zeros((3, N, N, N), dtype=cp.float64) # Global shape mask for the viewer

def add_dynamic_source(shape_name, behaviour_name, axis, position, start_step=0):
    """Injects a new dynamic source into the simulation."""
    global J_shape, current_sources
    
    # Calculate the time offset for this specific source
    start_t = (start_step + 0.5) * dt
    start_tau = start_t * c / dx
    
    # Generate Spatial Array
    if shape_name == "Current loop":
        shape_arr = sources.SourceShapes.current_loop(dx=dx, axis=axis, room_length=N, source_position=position)
    elif shape_name == "Dipole":
        shape_arr = sources.SourceShapes.dipole(axis=axis, room_length=N, source_position=position)
    else:
        return
        
    # Generate Temporal Lambda with time offset applied
    if behaviour_name == "Gaussian Pulse":
        beh_func = lambda t: sources.TemporalBehaviours.gaussian_pulse(t - start_tau)
    elif behaviour_name == "Continuous Wave":
        beh_func = lambda t: sources.TemporalBehaviours.continuous_wave(t - start_tau)
    elif behaviour_name == "Packet Burst":
        beh_func = lambda t: sources.TemporalBehaviours.packet_burst(t - start_tau)
    else:
        return
        
    gpu_shape = cp.asarray(shape_arr)
    
    current_sources.append({
        "gpu_shape": gpu_shape,
        "temporal behaviour": beh_func
    })
    
    # Accumulate into the viewer's diagnostic mask
    J_shape += cp.abs(gpu_shape)


# Initialize default starting sources
add_dynamic_source("Dipole", "Packet Burst", 1, (30, 50, 50))
add_dynamic_source("Dipole", "Packet Burst", 2, (70, 50, 50))


def step(n):
    """Advance the fields by one time step."""
    global E, B, H

    t = (n + 0.5) * dt
    tau = t * c / dx

    J = cp.zeros_like(E)
    for src in current_sources:
        pulse = src["temporal behaviour"](tau)
        J += eps_fixed * c * pulse * src["gpu_shape"]

    # Faraday's law: update B using the current E.
    B -= dt * curl(E, forward)
    H = B / mu
    # Ampere-Maxwell law: update E using the new B.
    E += dt * (curl(H, backward) - J) / eps


def check_constraints():
    """Return the divergence diagnostics used by the original script."""
    div_e = cp.max(cp.abs(divergence(eps*E, backward)))
    div_b = cp.max(cp.abs(divergence(B, forward)))
    div_j = cp.max(cp.abs(divergence(J_shape, backward)))
    return (
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