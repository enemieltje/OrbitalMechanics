import logging
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
