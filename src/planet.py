import logging

from matplotlib import pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


class Planet:
    mass: float
    color: str
    radius: float
    orbit_period: float
    equatorial_velocity: float
    gravitational_parameter: float
    oblateness_coefficient: float

    @property
    def J2(self):
        return self.oblateness_coefficient

    @property
    def mu(self):
        return self.gravitational_parameter

    def __init__(self, **kwargs) -> None:
        self.mass = kwargs.get("mass", 0)
        self.color = kwargs.get("color", "gray")
        self.radius = kwargs.get("radius", 0)
        self.orbit_period = kwargs.get("orbit_period", 0)
        self.equatorial_velocity = kwargs.get("equatorial_velocity", 0)
        self.oblateness_coefficient = kwargs.get("J2", 0)
        self.gravitational_parameter = self.mass * 6.67E-11
        logger.info(self.mu)

    def old_sun_synchronous_inclination(self, altitude: float) -> float:

        # Orbital radius (altitude + planet radius)
        orbit_radius = self.radius + altitude

        # Mean motion (n) - derived from Kepler's third law
        mean_motion = np.sqrt(
            self.gravitational_parameter / (orbit_radius ** 3))

        # Precession rate required for a sun-synchronous orbit (1 revolution per year)
        precession_rate = (2 * np.pi) / self.orbit_period

        # Calculate cos(i) for sun-synchronous orbit
        cos_i = precession_rate / (
            (-3/2) * self.oblateness_coefficient *
            (self.radius / orbit_radius) ** 2 *
            mean_motion
        )

        # Make sure cos(i) is within valid bounds for arccos
        cos_i = np.clip(cos_i, -1.0, 1.0)

        # Return the inclination in radians
        inclination = np.arccos(cos_i)

        return inclination

    def sun_synchronous_inclination(self, altitude: float) -> float:

        orbit_height = (self.radius + altitude)

        # Calculate cos(i)
        cos_i = \
            (-2/3) * \
            ((2 * np.pi) / self.orbit_period) * \
            (1 / self.oblateness_coefficient) * \
            ((orbit_height/self.radius)**2) * \
            np.sqrt(
                (orbit_height**3) /
                self.gravitational_parameter
            )

        # Make sure the value is within the valid range for arccos
        if cos_i < -1 or cos_i > 1:
            raise ValueError(
                "cos(i) is out of bounds. Check input values.")

        # Calculate the inclination i in radians
        inclination = np.acos(cos_i)

        return inclination

    def sun_synchronous_graph(self):
        # Generate altitude values (for example, from 200 km to 2000 km)
        # 500 points between 200 km and 1000 km
        altitudes = np.linspace(200E3, 1000E3, 500)

        # Calculate inclination for each altitude using the function
        inclinations = [self.sun_synchronous_inclination(
            altitude) for altitude in altitudes]

        # Plot the graph
        plt.figure(figsize=(8, 6))
        plt.plot(altitudes, np.degrees(inclinations),
                 label='Sun-Synchronous Inclination')
        plt.title('Sun-Synchronous Inclination vs Altitude')
        plt.xlabel('Altitude (m)')
        plt.ylabel('Inclination (degrees)')
        plt.grid(True)
        plt.legend()
        # plt.show()
        pass
