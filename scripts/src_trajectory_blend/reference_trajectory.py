import numpy as np
import ml_collections

from pydrake.common.value import Value
from pydrake.systems.framework import (
    LeafSystem,
    PublishEvent,
    TriggerType,
    BasicVector_,
)


class FigureEight(LeafSystem):
    def __init__(self, config: ml_collections.ConfigDict()):
        LeafSystem.__init__(self)

        # Class Parameters:
        self._UPDATE_RATE = config.reference_trajectory_rate

        # Initialize Abstract States:
        state_size = np.zeros((2,))
        state_init = Value[BasicVector_[float]](state_size)
        self.state_index = self.DeclareAbstractState(state_init)

        # Declare Output: Vector
        self.DeclareVectorOutputPort(
            "target_position",
            np.size(state_size),
            self.output_callback,
            {self.abstract_state_ticket(self.state_index)},
        )

        # Declare Initialization Event:
        def on_initialize(context, event):
            _x, _y = self._figure_eight_trajectory(context)
            a_state = context.get_mutable_abstract_state(self.state_index)
            a_state.set_value(np.array([_x, _y], dtype=float))

        self.DeclareInitializationEvent(
            event=PublishEvent(
                trigger_type=TriggerType.kInitialization,
                callback=on_initialize,
                )
            )

        # Declare Update Event: Current Trajectory
        def on_periodic(context, event):
            _x, _y = self._figure_eight_trajectory(context)
            a_state = context.get_mutable_abstract_state(self.state_index)
            a_state.set_value(np.array([_x, _y], dtype=float))

        self.DeclarePeriodicEvent(
            period_sec=self._UPDATE_RATE,
            offset_sec=0.0,
            event=PublishEvent(
                trigger_type=TriggerType.kPeriodic,
                callback=on_periodic,
                )
            )

    # Output Port Callback:
    def output_callback(self, context, output_target_position):
        # How can I clean this up?
        a_state = context.get_mutable_abstract_state(self.state_index)
        a_value = a_state.get_mutable_value()
        output_target_position.SetFromVector(a_value.get_mutable_value())

    # Class Methods:
    def _figure_eight_trajectory(self, context):
        _r = 1.0
        _time = context.get_time()
        _x = _r * np.cos(_time - np.pi / 2.0)
        _y = _r / 2.0 * np.sin(2 * _time)
        return _x, _y


# Test Instantiation:
if __name__ == "__main__":
    unit_test = FigureEight()
