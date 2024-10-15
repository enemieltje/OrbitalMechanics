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
        return np.linalg.norm(self.position_vector)

    @property
    def position_vector(self):

        # Extract Keplerian elements
        a = self.semi_major_axis
        e = self.eccentricity
        i = self.inclination
        Omega = self.right_ascension
        omega = self.argument_of_periapsis
        theta = self.true_anomaly

        # Step 1: Calculate distance from the focus using semi-major axis and eccentricity
        r = (a * (1 - e ** 2)) / \
            (1 + (e * np.cos(theta)))

        # Step 2: Position in the orbital plane
        x0 = r * np.cos(theta)
        y0 = r * np.sin(theta)
        z0 = 0  # Since it's in the orbital plane

        # Step 3: Transformation to the inertial frame

        # Rotation by argument of periapsis (around z-axis)
        x1 = x0 * np.cos(omega) - y0 * np.sin(omega)
        y1 = x0 * np.sin(omega) + y0 * np.cos(omega)
        z1 = z0  # No change in z in this step

        # Rotation by inclination (around x-axis)
        x2 = x1
        y2 = y1 * np.cos(i) - z1 * np.sin(i)
        z2 = y1 * np.sin(i) + z1 * np.cos(i)

        # Rotation by RAAN (around z-axis again)
        x = x2 * np.cos(Omega) - y2 * np.sin(Omega)
        y = x2 * np.sin(Omega) + y2 * np.cos(Omega)
        z = z2  # No change in z in this step

        return np.array([x, y, z])

    @property
    def velocity(self):
        if self._velocity:
            return self._velocity
        else:
            two_on_r = 2 / self.r
            one_on_a = 1 / self.a

            factor = two_on_r - one_on_a
            v_squared = self.planet.mu * factor

            return np.sqrt(v_squared)

    @velocity.setter
    def velocity(self, velocity):
        if not velocity:
            return
        self._velocity = velocity
        logger.warning(self.r)

        # Vis-visa equation
        self.semi_major_axis = 1 /\
            (
                (2 / self.r) -
                ((velocity ** 2) / self.planet.mu)
            )
        logger.warning(self.semi_major_axis)

        # eccentricity vector
        self.eccentricity = \
            ((velocity ** 2) / self.planet.mu) - \
            (1 / (self.r))
        logger.warning(self.eccentricity)

    @property
    def velocity_vector(self):
        pass

    def __str__(self) -> str:
        return (f"{CYAN}{self.name}{RESET}\n"
                f"Satellite Orbit:\n{super().__str__()}\n"
                f"Elapsed Time: {RED}{self.time}{RESET} s,\n"
                f"Mean Anomaly: {RED}{self.mean_anomaly:.4f}{RESET} radians,\n"
                f"Eccentric Anomaly: {RED}{
                    self.eccentric_anomaly:.4f}{RESET} radians,\n"
                f"True Anomaly: {RED}{self.true_anomaly:.4f}{RESET} radians\n"
                f"Distance to Planet: {RED}{self.r:.0f}{RESET} m,\n"
                f"velocity: {RED}{self.velocity:.0f}{RESET} m/s,\n"
                f"Position: {self.position_vector},\n"
                )

    def __init__(self, *args, **kwargs) -> None:
        self.name = kwargs.get("name", "Unnamed Sattelite")
        super().__init__(self, **kwargs)
        # self.orbit = orbit

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
        """Generalized plot function for orbit visualization."""
        if ax is None:
            fig, ax = plt.subplots()
        ax.set_aspect('equal')
        ax.scatter(x, y, label=plotlabel, color=color)
        ax.set_title(title)
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
        return self._plot_orbit(x, y, plotlabel=f"Step Position {self.name}", title=title, ax=ax, color=color)

    def kepler_graph(self, steps=100, ax=None, color='#ADB6C4'):
        x = np.zeros(steps)
        y = np.zeros(steps)

        for i, t in enumerate(np.linspace(0, self.period, steps)):
            self.time = t
            position = self.position_vector
            x[i] = position[0]
            y[i] = position[1]

        title = f'Orbit Propagation of {self.name}'
        return self._plot_orbit(x, y, plotlabel=f"Kepler Position {self.name}", title=title, ax=ax, color=color)

    def plot_combined(self, steps_step_graph=100000, steps_kepler_graph=100):
        logger.info(self)
        """Plot both step and Kepler graphs on the same figure."""
        # fig, axes = plt.subplots(1, 2, figsize=(
        #     12, 6))  # Two side-by-side subplots

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        # Plot step graph on the first subplot
        self.step_graph(steps=steps_step_graph, ax=ax)

        # Plot kepler graph on the second subplot
        self.kepler_graph(steps=steps_kepler_graph, ax=ax)

    def launch(self, altitude):
        logger.info(self)
        new_satellite = Satellite(name=self.name,
                                  planet=self.planet,
                                  altitude=altitude,
                                  )
        delta_v = abs(new_satellite.velocity - self.velocity)
        new_satellite.add_delta_v(delta_v)
        logger.info(f"{CYAN}{self.name}{RESET}\n"
                    f"Launching to {RED}{(altitude/1000):.0f}{RESET} km\n"
                    f"Velocity before launch: {RED}{
                        self.velocity:.0f}{RESET} m/s\n"
                    f"Velocity after launch: {RED}{
                        new_satellite.velocity:.0f}{RESET} m/s\n"
                    f"Expended Delta v: {RED}{delta_v:.0f}{RESET} m/s\n"
                    )

        return new_satellite

    def hohmann(self, altitude):
        # set position to periapsis
        self.time = 0
        self.change_apsis(altitude)

        # set position to apoapsis
        self.time = self.period/2
        self.change_apsis(altitude)

    def change_apsis(self, altitude):
        logger.info(self)
        # vis viva
        # a = average(where we are, where we want to be)
        v0 = self.velocity
        two_on_r = 2 / self.r
        one_on_a = 1 /\
            ((self.r +
              self.planet.radius + altitude) / 2)
        self.velocity = np.sqrt(self.planet.mu * (two_on_r - one_on_a))
        delta_v = abs(self.velocity - v0)
        self.add_delta_v(delta_v)

        logger.info(f"{CYAN}{self.name}{RESET}\n"
                    f"changing apsis to {RED}{
                        (altitude/1000):.0f}{RESET} km\n"
                    f"Velocity before launch: {RED}{
                        v0:.0f}{RESET} m/s\n"
                    f"Velocity after launch: {RED}{
                        self.velocity:.0f}{RESET} m/s\n"
                    f"Expended Delta v: {RED}{delta_v:.0f}{RESET} m/s\n"
                    )

        pass

    def add_delta_v(self, delta_v):
        self.delta_v += delta_v
