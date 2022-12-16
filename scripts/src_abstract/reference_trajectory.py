import numpy as np
import time

import pdb

from pydrake.systems.framework import LeafSystem, PublishEvent, TriggerType


class FigureEight(LeafSystem):
    def __init__(self):
        LeafSystem.__init__(self)
        
        # Class Parameters:
        self._UPDATE_RATE = 1.0 / 10.0 # MATCH MOTION PLANNER INPUT

        # Declare States:
        self.d_state_index= self.DeclareDiscreteState(2)

        # Declare Output:
        self.DeclareStateOutputPort("target_position", self.d_state_index)
        
        # Declare Initialization Event:
        def on_discrete_initialize(context, discrete_state):
            _trajectory_array = self._figure_eight_trajectory(context)
            discrete_state.get_mutable_vector().SetFromVector(_trajectory_array)

        self.DeclareInitializationDiscreteUpdateEvent(
            update=on_discrete_initialize,
            )

        # Declare Update Event:
        def on_periodic(context, discrete_state):
            _trajectory_array = self._figure_eight_trajectory(context)
            discrete_state.get_mutable_vector().SetFromVector(_trajectory_array)

        self.DeclarePeriodicDiscreteUpdateEvent(
            period_sec=self._UPDATE_RATE,
            offset_sec=0.0,
            update=on_periodic,  
            )

    # Methods:
    def _figure_eight_trajectory(self, context):
        _r = 1.0
        _time = context.get_time()
        _x = _r * np.cos(_time - np.pi / 2.0)
        _y = _r / 2.0 * np.sin(2 * _time)
        return np.array([_x, _y], dtype=float)