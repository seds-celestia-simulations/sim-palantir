"""Minimal 3D Yee-FDTD physics, with no visualization."""

import numpy as np
import sources

# Normalized units: c = epsilon0 = mu0 = 1.
N = 100
dx = dy = dz = 1.0
spacing = (dx, dy, dz)
dt = 0.5  # CFL limit for this grid: dt <= 1 / sqrt(3).

# Each field is one array: [component, x, y, z].
E = np.zeros((3, N, N, N))  # Ex, Ey, Ez
B = np.zeros((3, N, N, N))  # Bx, By, Bz


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


current_sources = (
    {
        "source shape": sources.SourceShapes.dipole(
                    axis=1,
                    room_length=N,
                    source_position=(30, 50, 50),
                    dipole_length=10,
                    maximum_signal_strength=10.0
                ),
        "temporal behaviour": lambda t: sources.TemporalBehaviours.gaussian_pulse(t, start_time=20.0)
    },
    {
        "source shape": sources.SourceShapes.dipole(
            axis=2,
            room_length=N,
            source_position=(70, 50, 50),
            dipole_length=10,
            maximum_signal_strength=10.0
        ),
        "temporal behaviour": lambda t: sources.TemporalBehaviours.continuous_wave(t, time_constant=5.0, angular_frequency=0.6)
    }
)


J_shape = np.zeros((3, N, N, N))
for current_source in current_sources: J_shape += current_source["source shape"]


def step(n):
    """Advance the fields by one time step."""
    global E, B

    t = (n + 0.5) * dt

    J = np.zeros((3, N, N, N))
    for current_source in current_sources: J += current_source["source shape"] * current_source["temporal behaviour"](t)

    # Faraday's law: update B using the current E.
    B -= dt * curl(E, forward)

    # Ampere-Maxwell law: update E using the new B.
    E += dt * (curl(B, backward) - J)


def check_constraints():
    """Return the divergence diagnostics used by the original script."""
    div_e = np.max(np.abs(divergence(E, backward)))
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
