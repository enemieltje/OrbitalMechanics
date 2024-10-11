import logging

from planet import Planet

logger = logging.getLogger(__name__)


class Orbit:
    semi_major_axis: float
    eccentricity: float
    inclination: float
    right_ascension: float
    argument_of_perigee: float
    true_anomaly: float

    def __init__(self, *args, **kwargs) -> None:

        pass
