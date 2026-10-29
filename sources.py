from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class SourceShapes:
    """Spatial arrangements for the current source. (values of J_shape)"""

    @staticmethod
    def current_loop(
        dx: float,
        axis: int,
        room_length: int,
        source_position: tuple[int, int, int] | None = None,
        maximum_signal_strength: float = 10.0
    ) -> NDArray[np.float64]:
        """A circular current loop oriented along one of the coordinate axes.

        Args:
            dx: The length of one grid cell in the simulation
            axis: The axis along which the dipole should be oriented. (0: x, 1: y, 2: z).
            room_length: The length of the simulation container.
            source_position: A tuple of the form (x, y, z) representing the position of the source. Defaults to the centre.
            maximum_signal_strength: The maximum amplitude of current density (default: 10.0).

        Returns:
            An array of shape (3, room_length, room_length, room_length).
        """
        if source_position is None: source_position = (room_length // 2, room_length // 2, room_length // 2)

        x, y, z = np.ogrid[: room_length, : room_length, : room_length]
        A = np.zeros((3, room_length, room_length, room_length))
        A[axis] = np.exp(- ((x - source_position[0]) ** 2 + (y - source_position[1]) ** 2 + (z - source_position[2]) ** 2) / 8)

        def backward(function: NDArray[np.float64], axis_index: int) -> NDArray[np.float64]:
            return (function - np.roll(function, 1, axis=axis_index)) / dx

        Fx, Fy, Fz = A
        return np.stack([
            backward(Fz, 1) - backward(Fy, 2),
            backward(Fx, 2) - backward(Fz, 0),
            backward(Fy, 0) - backward(Fx, 1),
        ]) * maximum_signal_strength

    @staticmethod
    def dipole(
        axis: int,
        room_length: int,
        source_position: tuple[int, int, int] | None = None,
        dipole_length: int = 12,
        maximum_signal_strength: float = 10.0
    ) -> NDArray[np.float64]:
        """A current source shaped like a magnetic dipole, oriented along one of the coordinate axes.

        Args:
            axis: The axis along which the dipole should be oriented. (0: x, 1: y, 2: z)
            room_length: The length of the simulation container.
            source_position: A tuple of the form (x, y, z) representing the position of the source. Defaults to the centre.
            dipole_length: The length of the current source (default: 12)
            maximum_signal_strength: The maximum amplitude of current density (default: 10.0)

        Returns:
            A 3D array of shape (3, room_length, room_length, room_length).
        """
        if source_position is None: source_position = (room_length // 2, room_length // 2, room_length // 2)

        J_shape = np.zeros((3, room_length, room_length, room_length))

        x, y, z = np.ogrid[: room_length, : room_length, : room_length]
        source_x, source_y, source_z = source_position
        match axis:
            case 0:
                r_parallel = x - source_x
                r_perpendicular_1 = y - source_y
                r_perpendicular_2 = z - source_z
            case 1:
                r_parallel = y - source_y
                r_perpendicular_1 = x - source_x
                r_perpendicular_2 = z - source_z
            case 2:
                r_parallel = z - source_z
                r_perpendicular_1 = x - source_x
                r_perpendicular_2 = y - source_y
        half_dipole_length = dipole_length / 2.0
        
        J_shape[axis] = maximum_signal_strength * np.exp(-(r_perpendicular_1 ** 2 + r_perpendicular_2 ** 2) / 0.1) * np.where(
            np.abs(r_parallel) <= half_dipole_length,
            0.5 * (1.0 + np.cos(np.pi * r_parallel / half_dipole_length)),
            0.0
        )
        return J_shape


class TemporalBehaviours:
    """Functions according to which the current source will vary with time."""

    @staticmethod
    def gaussian_pulse(
        t: float,
        start_time: float = 0.0,
        standard_deviation: float = 6.0,
        initial_phase: float = 0.0,
        angular_frequency: float = 0.6
    ) -> NDArray[np.float64]:
        """sin(omega * t) * e^(- ((t - t_0)/sigma)^2

        Args:
            t: time
            start_time: time offset (t_0) (default: 25.0)
            standard_deviation: the standard deviation (sigma) of the pulse function (default: 6.0)
            initial_phase: The initial phase of the sine function (default: 0.0)
            angular_frequency: The angular frequency (omega) of the sine function (default: 0.6)

        Returns:
            Amplitude array at time t
        """
        return np.sin(angular_frequency * t + initial_phase) * np.exp(-((t - start_time) / standard_deviation) ** 2)

    @staticmethod
    def ramped_up_sine(
        t: float,
        time_constant: float = 5.0,
        initial_phase: float = 0.0,
        angular_frequency: float = 0.6
    ) -> NDArray[np.float64]:
        """(1 - e^(-t/tau)) * sin(omega * t)

        Args:
            t: time
            time_constant: Time constant (tau) for ramping up the signal strength gradually (default: 5.0).
            initial_phase: The initial phase of the sine function (default: 0.0)
            angular_frequency: The angular frequency (omega) of the sine function (default: 0.6)

        Returns:
            Amplitude array at time t.
        """

        return (1.0 - np.exp(- t / time_constant)) * np.sin(angular_frequency * t + initial_phase)

    @staticmethod
    def packet_burst(
        t: float,
        burst_interval: tuple[float, float] = (10.0, 80.0),
        initial_phase: float = 0.0,
        angular_frequency: float = 0.6
    ) -> NDArray[np.float64]:
        """sin(omega * t) during some interval
        0 otherwise

        Args:
            t: time
            burst_interval: a tuple of the form (burst_start_time, burst_end_time). (default: 10.0, 80.0)
            initial_phase: The initial phase of the sine function (default: 0.0)
            angular_frequency: The angular frequency (omega) of the sine function (default: 0.6)

        Returns:
            Amplitude array at time t
        """
        return np.where((t >= burst_interval[0]) & (t <= burst_interval[1]), 1.0, 0.0) * np.sin(angular_frequency * t + initial_phase)