import numpy as np

from pydrake.common.value import Value
from pydrake.systems.framework import (
    LeafSystem,
    PublishEvent,
    TriggerType,
    BasicVector_,
)

import pdb

 
class DroneDynamics(LeafSystem):
    def __init__(self):
        LeafSystem.__init__(self)
        # Parameter Macros:
        self._IXX = 2.3951 * (10 ** (-5))
        self._IYY = 2.3951 * (10 ** (-5))
        self._IZZ = 3.2347 * (10 ** (-5))
        self._KM = 1.8580 * (10 ** (-5))
        self._KF = 0.005022