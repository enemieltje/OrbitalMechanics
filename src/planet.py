import logging

from matplotlib import pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


class Planet:
    mass: float
    radius: float
    period: float
    gravitational_parameter: float
    oblateness_coefficient: float

    def __init__(self, mass: float, radius: float, period: float, oblateness_coefficient: float) -> None:
        self.mass = mass
        self.radius = radius
        self.period = period
        self.oblateness_coefficient = oblateness_coefficient
        self.gravitational_parameter = self.mass * 6.67E-11

    def old_sun_synchronous_inclination(self, altitude: float) -> float:

        # Orbital radius (altitude + planet radius)
        orbit_radius = self.radius + altitude

        # Mean motion (n) - derived from Kepler's third law
        mean_motion = np.sqrt(
            self.gravitational_parameter / (orbit_radius ** 3))

        # Precession rate required for a sun-synchronous orbit (1 revolution per year)
        precession_rate = (2 * np.pi) / self.period

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
            ((2 * np.pi) / self.period) * \
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
        plt.show()
        pass
