import logging

from orbit import Orbit

logger = logging.getLogger(__name__)


class Satellite:
    orbit: Orbit = None
