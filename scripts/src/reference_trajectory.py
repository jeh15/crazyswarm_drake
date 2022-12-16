import numpy as np
import time

from pydrake.systems.framework import LeafSystem, PublishEvent, TriggerType



class FigureEight(LeafSystem):
    def __init__(self):
        LeafSystem.__init__(self)
        
        # Class Parameters:
        self._UPDATE_RATE = 1.0 / 5.0 # MATCH MOTION PLANNER INPUT

        # Initialize Output Values:
        self.trajectory = np.zeros((2,), dtype=float)

        # Declare Output: Trajectory Info
        """Outputs reference trajectory"""
        self.DeclareVectorOutputPort("reference_trajectory", 
        2, 
        self.output,
        {self.nothing_ticket()})
        
        # Declare Initialization Event:
        def on_initialize(context, event):
            _x, _y = self._figure_eight_trajectory(context, event)
            self.trajectory = np.array([_x, _y], dtype=float)

        self.DeclareInitializationEvent(
            event=PublishEvent(
                trigger_type=TriggerType.kInitialization,
                callback=on_initialize,
                )
            )

        # Declare Update Event: Current Trajectory
        def on_periodic(context, event):
            _x, _y = self._figure_eight_trajectory(context, event)
            self.trajectory = np.array([_x, _y], dtype=float)

        self.DeclarePeriodicEvent(period_sec=self._UPDATE_RATE,
                    offset_sec=0.0,
                    event=PublishEvent(
                        trigger_type=TriggerType.kPeriodic,
                        callback=on_periodic,
                        )
                    )

    # Output Port Callback:
    def output(self, context, reference_trajectory):
        reference_trajectory.SetFromVector(self.trajectory)

    # Methods:
    def _figure_eight_trajectory(self, context, event):
        _r = 1.0
        _time = context.get_time()
        _x = _r * np.cos(_time - np.pi / 2.0)
        _y = _r / 2.0 * np.sin(2 * _time)
        return _x, _y