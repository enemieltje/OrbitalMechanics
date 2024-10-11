import logging

logger = logging.getLogger(__name__)


class Planet:
    mass = None
    radius = None

    def __init__(self, mass: int, radius: int) -> None:
        self.mass = mass
        self.radius = radius

    def sun_synchronous_altitude(self, inclination: float):
        pass
