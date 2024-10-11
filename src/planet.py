import logging

logger = logging.getLogger(__name__)


class Planet:
    mass: float
    radius: float
    gravitational_parameter: float

    def __init__(self, mass: int, radius: int) -> None:
        self.mass = mass
        self.radius = radius
        self.gravitational_parameter = self.mass * 6.67E-11

    def sun_synchronous_altitude(self, inclination: float):
        pass
