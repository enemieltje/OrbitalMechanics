import logging
import numpy as np
import numpy.typing as npt

from planet import Planet

logger = logging.getLogger(__name__)

Vector = npt.NDArray[np.float64]

RESET = '\033[0m'
RED = '\033[31m'


class Orbit:

    # Kepler elements
    _semi_major_axis: float
    _eccentricity: float
    inclination: float
    right_ascension: float
    argument_of_periapsis: float

    # Other orbit elements
    planet: Planet
    semi_latus_rectum: float
    period: float
    periapsis: float
    apoapsis: float

    @property
    def semi_major_axis(self):
        return self._semi_major_axis

    @semi_major_axis.setter
    def semi_major_axis(self, semi_major_axis):
        self._semi_major_axis = semi_major_axis

    @property
    def eccentricity(self):
        return self._eccentricity

    @eccentricity.setter
    def eccentricity(self, eccentricity):
        self._eccentricity = eccentricity

    @property
    def altitude(self):
        return self._altitude

    @altitude.setter
    def altitude(self, altitude):
        self._altitude = altitude

    @property
    def semi_latus_rectum(self):
        return self.semi_major_axis * (1 - self.eccentricity ** 2)

    @property
    def periapsis(self):
        return self.semi_major_axis * (1 - self.eccentricity)

    @property
    def apoapsis(self):
        return self.semi_major_axis * (1 + self.eccentricity)

    @property
    def period(self):
        return 2 * np.pi * np.sqrt(
            (self.semi_major_axis ** 3) /
            (self.planet.gravitational_parameter)
        )

    @property
    def a(self):
        return self.semi_major_axis

    @property
    def e(self):
        return self.eccentricity

    @property
    def i(self):
        return self.inclination

    @property
    def Omega(self):
        return self.right_ascension

    @property
    def omega(self):
        return self.argument_of_periapsis

    def __str__(self) -> str:
        return (f"Semi-Major Axis: {RED}{self.semi_major_axis:.0f}{RESET} m,\n"
                f"Eccentricity: {RED}{self.eccentricity:.2f}{RESET},\n"
                f"Inclination: {RED}{self.inclination:.4f}{RESET} radians,\n"
                f"Right Ascension: {RED}{
                    self.right_ascension:.4f}{RESET} radians,\n"
                f"Argument of Periapsis: {RED}{
                    self.argument_of_periapsis:.4f}{RESET} radians,\n"
                f"Semi Latus Rectum: {RED}{
                    self.semi_latus_rectum:.0f}{RESET} m,\n"
                f"Periapsis: {RED}{
                    self.periapsis:.0f}{RESET} m,\n"
                f"Apoapsis: {RED}{
                    self.apoapsis:.0f}{RESET} m,\n"
                f"Period: {RED}{
                    self.period:.0f}{RESET} m,\n"
                )

    # create new orbit

    def __init__(self, *args, **kwargs) -> None:
        self.planet = kwargs.get("planet", None)
        self.semi_major_axis = kwargs.get("semi_major_axis", None)
        self.eccentricity = kwargs.get("eccentricity", None)
        self.inclination = kwargs.get("inclination", 0)
        self.right_ascension = kwargs.get("right_ascension", 0)
        self.argument_of_periapsis = kwargs.get("argument_of_perigee", 0)
        velocity = kwargs.get("velocity", None)

        altitude = kwargs.get("altitude", None)

        # Calculate Eccentricity and Semi-Major axis if not given
        if self.planet and self.eccentricity and self.semi_major_axis:
            # logger.warning("eccentricity, semi_major_axis")
            pass
        # elif self.planet and altitude and self.velocity:
        #     logger.warning("altitude, velocity")
            # self.eccentricity = \
            #     ((velocity ** 2) / self.planet.gravitational_parameter) - \
            #     (1 / (self.planet.radius + altitude))
            # # logger.warning(self.eccentricity)
            # self.semi_major_axis = 1 / \
            #     ((2 / (self.planet.radius + altitude)) -
            #      ((velocity ** 2) / self.planet.gravitational_parameter))
            # pass

        elif self.planet and altitude:
            # start with circular orbit
            self.semi_major_axis = self.planet.radius + altitude
            self.eccentricity = 0
            # logger.warning(f"v: {self.velocity}")

            # change velocity to compensate if needed
            # self.velocity = velocity
            # logger.warning(f"semi_major_axis: {self.semi_major_axis}")

        elif self.planet:
            # logger.warning("none")
            # Landed on equator "orbit"
            self.semi_major_axis = self.planet.radius / 2
            self.eccentricity = 0.9999
            self.velocity = self.planet.equatorial_velocity

        else:
            raise ValueError("Invalid arguments for orbit")

    def to_inertial_plane(self, x0, y0, z0):
        i = self.inclination
        Omega = self.right_ascension
        omega = self.argument_of_periapsis

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
