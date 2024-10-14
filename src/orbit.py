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
    semi_major_axis: float
    eccentricity: float
    inclination: float
    right_ascension: float
    argument_of_periapsis: float

    # Other orbit elements
    planet: Planet
    semi_latus_rectum: float
    period: float
    periapsis: float
    apoapsis: float
    eccentricity_vector: Vector

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
                )

    # create new orbit

    def __init__(self, *args, **kwargs) -> None:
        self.planet = kwargs.get("planet", None)
        self.semi_major_axis = kwargs.get("semi_major_axis", None)
        self.eccentricity = kwargs.get("eccentricity", None)
        self.altitude = kwargs.get("altitude", None)
        self.velocity = kwargs.get("velocity", None)
        self.inclination = kwargs.get("inclination", 0)
        self.right_ascension = kwargs.get("right_ascension", 0)
        self.argument_of_periapsis = kwargs.get("argument_of_perigee", 0)

        # fill in remaining orbit elements
        if self.planet and self.semi_major_axis and self.eccentricity:
            self.fill_elements()

        elif self.planet and self.altitude and self.velocity:
            self.eccentricity = \
                ((self.velocity ** 2) / self.planet.gravitational_parameter) - \
                (1 / (self.planet.radius + self.altitude))
            self.semi_major_axis = 1 / \
                ((2 / (self.planet.radius + self.altitude)) -
                 ((self.velocity ** 2) / self.planet.gravitational_parameter))

            self.fill_elements()

        elif self.planet and self.altitude:
            # assume circular orbit
            self.semi_major_axis = self.planet.radius + self.altitude
            self.eccentricity = 0
            self.fill_elements()

        else:
            raise ValueError("Invalid arguments for orbit")

    def fill_elements(self):
        self.semi_latus_rectum = \
            self.semi_major_axis * (1 - self.eccentricity ** 2)
        self.periapsis = self.semi_major_axis * (1 - self.eccentricity)
        self.apoapsis = self.semi_major_axis * (1 + self.eccentricity)
        self.period = \
            2 * np.pi * np.sqrt(
                (self.semi_major_axis ** 3) /
                (self.planet.gravitational_parameter)
            )
