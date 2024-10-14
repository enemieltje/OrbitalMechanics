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

earth = Planet(mass=5.972E24,
               radius=6378E3,
               period=365.256363004 * 24 * 60 * 60,
               J2=1082.62668E-6)

mars = Planet(mass=6.4171E23,
              radius=3389E3,
              period=779.94 * 24 * 60 * 60,
              J2=1960.45E-6)

# Assignment 4
sat_1 = Satellite(name="Sat 1",
                  planet=earth,
                  semi_major_axis=2.5E7,
                  eccentricity=0.2)
sat_1.time = 2000
logger.info(sat_1)

sat_2 = Satellite(name="Sat 2",
                  planet=earth,
                  semi_major_axis=earth.radius + 700E3,
                  eccentricity=0.05)
sat_2.time = 3000
logger.info(sat_2)

logger.info(
    f"{RESET}Distance: {YELLOW}{np.linalg.norm(sat_2.position - sat_1.position):.0f}{RESET} m\n")


sat_3 = Satellite(name="Sat 3",
                  planet=mars,
                  semi_major_axis=2.5E7,
                  eccentricity=0.2)
sat_3.time = 2000
logger.info(sat_3)

sat_4 = Satellite(name="Sat 4",
                  planet=mars,
                  semi_major_axis=earth.radius + 700E3,
                  eccentricity=0.05)
sat_4.time = 3000
logger.info(sat_4)

logger.info(
    f"{RESET}Distance: {YELLOW}{np.linalg.norm(sat_4.position - sat_3.position):.0f}{RESET} m\n")


# Assignment 5
earth.sun_synchronous_graph()

moon = Satellite(name="Moon",
                 planet=earth,
                 semi_major_axis=384399E3,
                 eccentricity=0.0549,
                 inclination=np.radians(5.145))
logger.info(moon)
# logger.info(moon.apsides_precession())
logger.info(
    f"{RESET}Precession of Apsides: {YELLOW}{moon.apsides_precession()}{RESET} rad/rev\n")


# Assignment 6

IH_1 = Satellite(name="InhollandSat 1",
                 planet=earth,
                 altitude=525E3,
                 velocity=7728)
# TODO: Overlay these
IH_1.kepler_graph(10000)
IH_1.step_graph(10000)

plt.show()

IH_2 = Satellite(name="InhollandSat 2",
                 orbit=earth.surface_orbit())
IH_2.launch(525E3)
IH_2.hohmann(385000E3)
IH_2.incline(18)

IH_3 = Satellite(name="InhollandSat 3",
                 planet=earth,
                 altitude=525E3)

IH_3.incline(18)
IH_3.hohmann(385000E3)

mars_sat = Satellite(name="Mars Sat 1",
                     orbit=mars.surface_orbit())
mars_sat.launch(200)

logger.info(mars.stationary())

mars.sun_synchronous_graph()
