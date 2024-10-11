import logging
import sys

from orbit import Orbit
from planet import Planet
from satellite import Satellite

# Configure logs to log both in the console and to a file
logger = logging.getLogger(__name__)
logging.basicConfig(handlers=[logging.StreamHandler(sys.stdout)],
                    encoding='utf-8', level=logging.DEBUG)

earth = Planet()
mars = Planet()

# Assignment 4
sat_1 = Satellite(Orbit(planet=earth, semi_major_axis=2.5E7, eccentricity=0.2))
sat_1.step_time(2000)

sat_2 = Satellite(
    Orbit(planet=earth, semi_major_axis=earth.radius + 700E3, eccentricity=0.05))
sat_2.step_time(3000)

logger.info(sat_2.position - sat_1.position)


sat_3 = Satellite(Orbit(planet=mars, semi_major_axis=2.5E7, eccentricity=0.2))
sat_3.step_time(2000)

sat_4 = Satellite(
    Orbit(planet=mars, semi_major_axis=earth.radius + 700E3, eccentricity=0.05))
sat_4.step_time(3000)

logger.info(sat_4.position - sat_3.position)

# Assignment 5
earth.sun_synchronous_graph()

moon = Satellite(Orbit(planet=earth, altitude=None))
moon.apsides_precession()


# Assignment 6

IH_1 = Satellite(Orbit(altitude=525E3, velocity=7728))
# TODO: Overlay these
IH_1.orbit.kepler_graph()
IH_1.orbit.step_graph()


IH_2 = Satellite(earth.surface_orbit())
IH_2.launch(525E3)
IH_2.hohmann(385000E3)
IH_2.incline(18)

IH_3 = Satellite(Orbit(planet=earth, altitude=525E3))

IH_3.incline(18)
IH_3.hohmann(385000E3)

mars_sat = Satellite(mars.surface_orbit())
mars_sat.launch(200)

logger.info(mars.stationary())

mars.sun_synchronous_graph()