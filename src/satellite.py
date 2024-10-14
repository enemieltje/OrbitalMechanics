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
    _time: int = 0
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
    def position(self):

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

    def __str__(self) -> str:
        return (f"{CYAN}{self.name}{RESET}\n"
                f"Satellite Orbit:\n{super().__str__()}\n"
                f"Elapsed Time: {RED}{self.time}{RESET} s,\n"
                f"Mean Anomaly: {RED}{self.mean_anomaly:.4f}{RESET} radians,\n"
                f"Eccentric Anomaly: {RED}{
                    self.eccentric_anomaly:.4f}{RESET} radians,\n"
                f"True Anomaly: {RED}{self.true_anomaly:.4f}{RESET} radians\n"
                f"Position: {self.position},\n"
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

    def step_graph(self, steps=10000):
        # length of timestep to make one period
        dt = self.period/steps

        x = np.zeros(steps+1)
        y = np.zeros(steps+1)
        # (m) initial condition of object position on x-axis (1.496e11 is averagedistance Earth-Sun)
        x[0] = self.periapsis
        y[0] = 0.0  # (m) initial condition of object position on y-axis
        vx = np.zeros(steps+1)
        vy = np.zeros(steps+1)
        # (m/s) initial condition of object velocity in x-axis direction
        vx[0] = 0.0
        # (m/s) initial condition of object velocity in y-axis direction (Earth velocity around Sun is 29780.0 m/s)
        vy[0] = np.sqrt(self.planet.mu*((2/x[0])-(1/self.semi_major_axis)))
        ax = np.zeros(steps+1)
        ay = np.zeros(steps+1)

        def a(x, y):
            # Distance from the central body to the object
            r = np.sqrt(x**2+y**2)
            # Acceleration of a body under a gravity force from a mass m1
            return (-self.planet.gravitational_parameter)/r**2

        for i in range(0, steps):
            if x[i] == 0:
                # To prevent an impossible operations in the calculation of ax[i]
                ax[i] = 0
            else:
                # Calculation of the acceleration ax[i]. Note that the sign/direction of ax[i] is arranged with the factor: (x[i]/abs(x[i]))
                ax[i] = a(x[i], y[i])/np.sqrt(1+(y[i]**2/x[i]**2)) * \
                    (x[i]/abs(x[i]))
            if y[i] == 0:
                # To prevent an impossible operation in the calculation of ay[i]
                ay[i] = 0
            else:
                # Calculation of the acceleration ay[i]. Note that the sign/direction of ay[i] is arranged with the factor: (y[i]/abs(y[i]))
                ay[i] = a(x[i], y[i])/np.sqrt(1+(x[i]**2/y[i]**2)) * \
                    (y[i]/abs(y[i]))
            # calculation of the velocity as a result of the initial velocity and the acceleration
            vx[i+1] = vx[i] + ax[i]*dt
            # calculation of the velocity as a result of the initial velocity and the acceleration
            vy[i+1] = vy[i] + ay[i]*dt
            # calculation of the position as a result of the initial position and the velocity
            x[i+1] = x[i] + vx[i]*dt
            # calculation of the position as a result of the initial position and the velocity
            y[i+1] = y[i] + vy[i]*dt

        plotlabel = 'Object position'
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        fig.suptitle(f'Step Orbit propagation of {self.name}')
        plt.scatter(x, y, label=plotlabel)
        plt.legend(loc='upper left', fontsize='small')
        plt.xlabel('X axis (m)')
        plt.ylabel('Y axis (m)')
        # Yellow is used for the color of the Sun
        circle = plt.Circle((0, 0), radius=self.planet.radius, color='yellow')
        ax.add_patch(circle)
        # plt.show()

    def kepler_graph(self, steps=100):

        x = np.zeros(steps)
        y = np.zeros(steps)

        for i, t in enumerate(np.linspace(0, self.period, steps)):
            self.time = t
            position = self.position
            x[i] = position[0]
            y[i] = position[1]

        plotlabel = 'Object position'
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        fig.suptitle(f'Kepler Orbit propagation of {self.name}')
        plt.scatter(x, y, label=plotlabel)
        plt.legend(loc='upper left', fontsize='small')
        plt.xlabel('X axis (m)')
        plt.ylabel('Y axis (m)')
        # Yellow is used for the color of the Sun
        circle = plt.Circle((0, 0), radius=self.planet.radius, color='yellow')
        ax.add_patch(circle)
        # plt.show()
