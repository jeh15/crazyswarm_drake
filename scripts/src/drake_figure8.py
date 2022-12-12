import numpy as np
import time

from pydrake.systems.framework import LeafSystem, PublishEvent, TriggerType



class Trajectory_Figure8(LeafSystem):
    def __init__(self):
        LeafSystem.__init__(self)
        
        # Class Parameters:
        self._UPDATE_RATE = 1.0 / 500.0 # 500 Hz -> 2 ms

        # Declare Output: Trajectory Info
        """Outputs reference trajectory"""
        self.DeclareVectorOutputPort("trajectory", 3, self.output)
        
        # Declare Initialization Event:
        def on_initialize(context, event):
            self.trajectory = np.array([0.0, 0.0, 0.0], dtype=float)

        self.DeclareInitializationEvent(
            event=PublishEvent(
                trigger_type=TriggerType.kInitialization,
                callback=on_initialize,
                )
            )

        # Declare Update Event: Current Trajectory
        def on_periodic(context, event):
            _x, _y = self._figure_eight_trajectory(context, event)
            self.trajectory = np.array([_x, _y, 0.0], dtype=float)

        self.DeclarePeriodicEvent(period_sec=self._UPDATE_RATE,
                    offset_sec=0.0,
                    event=PublishEvent(
                        trigger_type=TriggerType.kPeriodic,
                        callback=on_periodic,
                        )
                    )

    # Output Port Callback:
    def output(self, context, get_trajectory):
        get_trajectory.SetFromVector(self.trajectory)

    # Methods:
    def _figure_eight_trajectory(self, context, event):
        _r = 1.0
        _time = context.get_time()
        _x = _r * np.cos(_time - np.pi / 2.0)
        _y = _r / 2.0 * np.sin(2 * _time)
        return _x, _y