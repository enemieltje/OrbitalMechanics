import logging
import sys
from matplotlib import pyplot as plt
import numpy as np
from colorlog import ColoredFormatter

from orbit import Orbit
from planet import Planet
from satellite import Satellite

# Configure logs with colors
RESET = '\033[0m'
RED = '\033[31m'
YELLOW = '\033[33m'
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(ColoredFormatter(
    log_colors={
        'DEBUG':    'cyan',
        'INFO':     'light_black',
        'WARNING':  'yellow',
        'ERROR':    'red',
        'CRITICAL': 'bold_red',
    }
))
logging.basicConfig(handlers=[handler],
                    encoding='utf-8', level=logging.INFO)
logger = logging.getLogger(__name__)

# Create plannets
earth = Planet(mass=5.972E24,
               radius=6371E3,
               orbit_period=365.256363004 * 24 * 60 * 60,
               equatorial_velocity=465,
               J2=1082.62668E-6,
               color="#294C60"
               )

mars = Planet(mass=6.4171E23,
              radius=3389E3,
              orbit_period=779.94 * 24 * 60 * 60,
              equatorial_velocity=241,
              J2=1960.45E-6,
              color="#FFC49B")

# Assignment 4
fig = plt.figure()
earth_ax = fig.add_subplot(121)
earth_ax.set_aspect('equal')
earth_ax.set_title("Orbit Propagation around Earth")
mars_ax = fig.add_subplot(122)
mars_ax.set_title("Orbit Propagation around Mars")
mars_ax.set_aspect('equal')

sat_1 = Satellite(name="Sat 1",
                  planet=earth,
                  semi_major_axis=2.5E7,
                  eccentricity=0.2)
sat_1.time = 2000
sat_1.kepler_graph(ax=earth_ax)
logger.info(sat_1)

sat_2 = Satellite(name="Sat 2",
                  planet=earth,
                  semi_major_axis=earth.radius + 700E3,
                  eccentricity=0.05)
sat_2.time = 3000
sat_2.kepler_graph(ax=earth_ax, color='#FFEFD3')
logger.info(sat_2)

logger.info(
    f"{RESET}Distance: {YELLOW}{np.linalg.norm(sat_2.position_vector - sat_1.position_vector):.0f}{RESET} m\n")


sat_3 = Satellite(name="Sat 3",
                  planet=mars,
                  semi_major_axis=2.5E7,
                  eccentricity=0.2)
sat_3.time = 2000
sat_3.kepler_graph(ax=mars_ax)
logger.info(sat_3)

sat_4 = Satellite(name="Sat 4",
                  planet=mars,
                  semi_major_axis=earth.radius + 700E3,
                  eccentricity=0.05)
sat_4.time = 3000
sat_4.kepler_graph(ax=mars_ax, color='#FFEFD3')
logger.info(sat_4)

logger.info(
    f"{RESET}Distance: {YELLOW}{np.linalg.norm(sat_4.position_vector - sat_3.position_vector):.0f}{RESET} m\n")


# Assignment 5
earth.sun_synchronous_graph()

moon = Satellite(name="Moon",
                 planet=earth,
                 semi_major_axis=384399E3,
                 eccentricity=0.0549,
                 inclination=np.radians(5.145))
logger.info(moon)
precession = moon.apsides_precession()

logger.info(
    f"{RESET}Precession of Apsides: {YELLOW}{precession}{RESET} rad/rev\n")


# Assignment 6
IH_1 = Satellite(name="InhollandSat 1",
                 planet=earth,
                 altitude=525E3
                 )
IH_1.change_velocity(7728)
IH_1.plot_combined(1000)

IH_2 = Satellite(name="InhollandSat 2",
                 planet=earth,
                 altitude=525E3)

IH_2.hohmann(385000E3)
IH_2.incline(np.radians(18))
logger.info(
    f"{RESET}Total Expended Delta v: {YELLOW}{(IH_2.delta_v):.0f}{RESET} km\n")

IH_3 = Satellite(name="InhollandSat 3",
                 planet=earth,
                 altitude=525E3)

IH_3.incline(np.radians(18))
IH_3.hohmann(385000E3)
logger.info(
    f"{RESET}Total Expended Delta v: {YELLOW}{(IH_3.delta_v):.0f}{RESET} km\n")

logger.info(IH_2)
logger.info(IH_3)


earth_sat = Satellite(name="Earth Sat", planet=earth)
earth_sat.launch(250E3)

mars_sat = Satellite(name="Mars Sat", planet=mars)
mars_sat.launch(200E3)

logger.info(
    f"{RESET}Altitude of Geostationary orbit: {YELLOW}{(earth.stationary/1000):.0f}{RESET} km\n")
logger.info(
    f"{RESET}Altitude of Areostationary orbit: {YELLOW}{(mars.stationary/1000):.0f}{RESET} km\n")

mars.sun_synchronous_graph()

plt.show()
