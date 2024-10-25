import logging
from matplotlib import pyplot as plt
import numpy as np
import numpy.typing as npt
from orbit import Orbit

logger = logging.getLogger(__name__)
Vector = npt.NDArray[np.float64]

RESET = '\033[0m'
RED = '\033[31m'
CYAN = '\033[36m'


class Satellite(Orbit):
    # orbit: Orbit = None
    name: str
    delta_v: float = 0
    _time: int = 0
    _velocity: float = 0
    _eccentric_anomaly: float = None

    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, time):
        self._time = time
        self.update_eccentric_anomaly()

    @property
    def mean_anomaly(self):
        return 2 * np.pi * (self._time / self.period)

    @property
    def eccentric_anomaly(self):
        if not self._eccentric_anomaly:
            self.update_eccentric_anomaly()
        return self._eccentric_anomaly

    @property
    def true_anomaly(self):
        return 2*np.atan(
            np.tan(self.eccentric_anomaly / 2) *
            np.sqrt(
                (1 + self.eccentricity) /
                (1 - self.eccentricity)
            )
        )

    @property
    def theta(self):
        return self.true_anomaly

    @property
    def r(self):
        # Orbit equation:
        return (self.a * (1 - self.e ** 2)) / \
            (1 + (self.e * np.cos(self.theta)))

    @property
    def position_vector(self):

        return self.to_inertial_plane(
            self.r * np.cos(self.theta),
            self.r * np.sin(self.theta),
            0
        )

    @property
    def velocity(self):
        # Vis-viva equation
        two_on_r = 2 / self.r
        one_on_a = 1 / self.a

        factor = two_on_r - one_on_a
        v_squared = self.planet.mu * factor

        return np.sqrt(v_squared)

    @velocity.setter
    def velocity(self, velocity):
        pass

    @property
    def velocity_vector(self):
        # flight path angle (from: wiki Elliptic Orbit)
        tan_flight_path_angle = (self.e * np.sin(self.true_anomaly)) / \
                                (1 + (self.e * np.cos(self.true_anomaly)))
        flight_path_angle = np.arctan(tan_flight_path_angle)
        velocity_angle = self.true_anomaly + np.pi / 2 - flight_path_angle

        return self.to_inertial_plane(
            self.velocity * np.cos(velocity_angle),
            self.velocity * np.sin(velocity_angle),
            0
        )

    def __str__(self) -> str:
        return (f"{CYAN}{self.name}{RESET}\n"
                f"Satellite Orbit:\n{super().__str__()}\n"
                f"Elapsed Time: {RED}{self.time}{RESET} s,\n"
                f"Mean Anomaly: {RED}{self.mean_anomaly:.2f}{RESET} radians,\n"
                f"Eccentric Anomaly: {RED}{
                    self.eccentric_anomaly:.2f}{RESET} radians,\n"
                f"True Anomaly: {RED}{self.true_anomaly:.2f}{RESET} radians\n"
                f"Distance to Planet: {RED}{self.r:.0f}{RESET} m,\n"
                f"velocity: {RED}{self.velocity:.0f}{RESET} m/s,\n"
                f"Expended Delta v: {RED}{self.delta_v:.0f}{RESET} m/s\n"
                f"Position: [{RED}{self.position_vector[0]:.0f}{RESET}, {RED}{
                    self.position_vector[1]:.0f}{RESET}, {RED}{self.position_vector[2]:.0f}{RESET}],\n"
                )

    def __init__(self, *args, **kwargs) -> None:
        self.name = kwargs.get("name", "Unnamed Sattelite")
        super().__init__(self, **kwargs)

    def _get_eccentric_anomaly(self, depth: int):
        if depth == 0:
            return 0
        eccentric_anomaly = self._get_eccentric_anomaly(depth - 1)
        return self.mean_anomaly - (self.eccentricity * np.sin(eccentric_anomaly))

    def update_eccentric_anomaly(self):
        self._eccentric_anomaly = self._get_eccentric_anomaly(10)

    def apsides_precession(self):
        # Precession of apsides formula:
        # Δω = 3π * (J2 * R_E^2) / p^2 * (2 - (5/2) * sin^2(i))

        # Calculate intermediate components
        J2_term = self.planet.J2 * (self.planet.radius ** 2)
        p_term = self.semi_latus_rectum ** 2
        sine_term = np.sin(self.i) ** 2
        factor = (2 - (5 / 2) * sine_term)

        # Combine the pieces
        delta_omega = 3 * np.pi * (J2_term / p_term) * factor

        return delta_omega

    def _plot_orbit(self, x, y, plotlabel, title, ax=None, color='blue'):
        # Generalized plot function for orbit visualization.
        if ax is None:
            fig, ax = plt.subplots()
        ax.set_aspect('equal')
        ax.scatter(x, y, label=plotlabel, color=color)

        ax.set_xlabel('X axis (m)')
        ax.set_ylabel('Y axis (m)')
        circle = plt.Circle((0, 0), radius=self.planet.radius,
                            color=self.planet.color)
        ax.add_patch(circle)
        ax.legend(loc='upper left', fontsize='small')
        return ax

    def step_graph(self, steps=100000, ax=None, color='#FFEFD3'):
        dt = self.period / steps

        x = np.zeros(steps + 1)
        y = np.zeros(steps + 1)
        x[0] = self.periapsis
        y[0] = 0.0
        vx = np.zeros(steps + 1)
        vy = np.zeros(steps + 1)
        vy[0] = np.sqrt(self.planet.mu * ((2 / x[0]) -
                        (1 / self.semi_major_axis)))
        ax_vals = np.zeros(steps + 1)
        ay_vals = np.zeros(steps + 1)

        def a(x, y):
            r = np.sqrt(x**2 + y**2)
            return -self.planet.gravitational_parameter / r**2

        for i in range(steps):
            if x[i] == 0:
                ax_vals[i] = 0
            else:
                ax_vals[i] = a(x[i], y[i]) / np.sqrt(1 +
                                                     (y[i]**2 / x[i]**2)) * (x[i] / abs(x[i]))
            if y[i] == 0:
                ay_vals[i] = 0
            else:
                ay_vals[i] = a(x[i], y[i]) / np.sqrt(1 +
                                                     (x[i]**2 / y[i]**2)) * (y[i] / abs(y[i]))

            vx[i + 1] = vx[i] + ax_vals[i] * dt
            vy[i + 1] = vy[i] + ay_vals[i] * dt
            x[i + 1] = x[i] + vx[i] * dt
            y[i + 1] = y[i] + vy[i] * dt

        title = f'Orbit Propagation of {self.name}'
        return self._plot_orbit(x, y, plotlabel=f"Step Orbit {self.name}", title=title, ax=ax, color=color)

    def kepler_graph(self, steps=100, ax=None, color='#ADB6C4'):
        x = np.zeros(steps)
        y = np.zeros(steps)

        initial_time = self.time

        for i, t in enumerate(np.linspace(0, self.period, steps)):
            self.time = t
            position = self.position_vector
            x[i] = position[0]
            y[i] = position[1]

        self.time = initial_time
        position = self.position_vector

        title = f'Orbit Propagation of {self.name}'

        self._plot_orbit(x, y, plotlabel=f"Kepler Orbit {
                         self.name}", title=title, ax=ax, color=color)
        self._plot_orbit(position[0], position[1], plotlabel=f"Position {
                         self.name}", title=title, ax=ax, color='magenta')

    def plot_combined(self, steps_step_graph=100000, steps_kepler_graph=100):
        logger.info(self)
        # Plot both step and Kepler graphs on the same figure.

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        ax.set_title(f"Orbit Propagation of {self.name}")
        # Plot step graph on the first subplot
        self.step_graph(steps=steps_step_graph, ax=ax)

        # Plot kepler graph on the second subplot
        self.kepler_graph(steps=steps_kepler_graph, ax=ax)

    def launch(self, altitude):
        logger.info(self)

        # velocity at equator
        v0 = self.planet.equatorial_velocity

        # Setting circular orbit after launch
        self.semi_major_axis = self.planet.radius + altitude
        self.eccentricity = 0

        # Calculating gravity loss
        g_surface = self.planet.g
        g_orbit = self.planet.mu / (self.semi_major_axis) ** 2
        g_avg = (g_surface + g_orbit) / 2

        gravity_loss = np.sqrt(2 * g_avg * altitude)

        # Calculating required delta v
        delta_v = abs(self.velocity + gravity_loss - v0)

        # logging results
        self.add_delta_v(delta_v)
        logger.info(f"{CYAN}{self.name}{RESET}\n"
                    f"Launching to {RED}{(altitude/1000):.0f}{RESET} km\n"
                    f"Velocity before launch: {RED}{
                        v0:.0f}{RESET} m/s\n"
                    f"Gravity loss: {RED}{
                        gravity_loss:.0f}{RESET} m/s\n"
                    f"Velocity after launch: {RED}{
                        self.velocity:.0f}{RESET} m/s\n"
                    f"Expended Delta v: {RED}{delta_v:.0f}{RESET} m/s\n"
                    )

    def hohmann(self, altitude):
        # set position to periapsis
        self.time = 0
        self.change_apsis(altitude)

        # set position to apoapsis
        self.time = self.period/2
        self.change_apsis(altitude)

    def change_velocity(self, velocity):
        # Check if at apsis
        if not (self.time == 0 or self.time == self.period/2):
            logger.error("Can only change velocity at an apsis!")
            return

        # Calculate delta v
        v0 = self.velocity
        delta_v = abs(v0 - velocity)
        r = self.r

        # Vis-viva equation
        self.semi_major_axis = 1 /\
            (
                (2 / r) -
                ((velocity ** 2) / self.planet.mu)
            )

        # Periapsis formula / simplified eccentricity vector
        eccentricity = 1.0 - (r / self.semi_major_axis)

        # Check if apoapsis and periapsis changed places
        if eccentricity > 0:
            # In periapsis after velocity change
            self.eccentricity = eccentricity
            self.time = 0
        else:
            # In apoapsis after velocity change
            self.eccentricity = -eccentricity
            self.time = self.period/2

        # log the result
        self.add_delta_v(delta_v)

        logger.info(f"{CYAN}{self.name}{RESET}\n"
                    f"Changing velocity to {RED}{
                        (velocity):.0f}{RESET} m/s\n"
                    f"Velocity before burn: {RED}{
                        v0:.0f}{RESET} m/s\n"
                    f"Expended Delta v: {RED}{delta_v:.0f}{RESET} m/s\n"
                    )

    def change_apsis(self, altitude):

        # Check if at apsis
        if not (self.time == 0 or self.time == self.period/2):
            logger.error("Can only change apsis at an apsis!")
            return

        v0 = self.velocity
        r = self.r

        # a = average(where we are, where we want to be)
        self.semi_major_axis = (r + self.planet.radius + altitude) / 2
        eccentricity = 1.0 - (r / self.semi_major_axis)

        # Check if apoapsis and periapsis changed places
        if eccentricity > 0:
            # In periapsis after velocity change
            self.eccentricity = eccentricity
            self.time = 0
        else:
            # In apoapsis after velocity change
            self.eccentricity = -eccentricity
            self.time = self.period/2

        delta_v = abs(self.velocity - v0)
        self.add_delta_v(delta_v)

        logger.info(f"{CYAN}{self.name}{RESET}\n"
                    f"Changing apsis to {RED}{
                        (altitude/1000):.0f}{RESET} km\n"
                    f"Velocity before burn: {RED}{
                        v0:.0f}{RESET} m/s\n"
                    f"Velocity after burn: {RED}{
                        self.velocity:.0f}{RESET} m/s\n"
                    f"Expended Delta v: {RED}{delta_v:.0f}{RESET} m/s\n"
                    )

    def incline(self, inclination):
        # Check if orbit is circular
        if abs(self.e) > 1E-10:
            logger.error(
                "Can only change inclination at when orbit is circular!")
            return

        # calculate inclination change
        delta_i = abs(self.inclination - inclination)
        self.inclination = inclination

        # calculate required delta v
        delta_v = 2*self.velocity * np.sin(delta_i/2)

        # log the results
        self.add_delta_v(delta_v)
        logger.info(f"{CYAN}{self.name}{RESET}\n"
                    f"Changing inclination to {RED}{
                        self.i:.2f}{RESET} radians\n"
                    f"Velocity: {RED}{
                        self.velocity:.0f}{RESET} m/s\n"
                    f"Expended Delta v: {RED}{delta_v:.4f}{RESET} m/s\n"
                    )

    def add_delta_v(self, delta_v):
        self.delta_v += delta_v
