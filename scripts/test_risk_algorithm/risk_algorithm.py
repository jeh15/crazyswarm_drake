import numpy as np
import ml_collections

import jax
import jax.numpy as jnp

from pydrake.common.value import Value
from pydrake.systems.framework import (
    LeafSystem,
    PublishEvent,
    TriggerType,
    BasicVector_,
)

from pycrazyswarm import *
import motioncapture

import pdb


class Adversary(LeafSystem):
    def __init__(self, config: ml_collections.ConfigDict()):
        LeafSystem.__init__(self)
        # Parameters:
        self._CONTROL_RATE = config.crazyswarm_rate  
        self._OUTPUT_UPDATE_RATE = config.motion_planner_rate

        # Initialize Vector:
        self._data = []

        """ Initialize Abstract States: (Output Only) """
        output_size = np.zeros((self._spline_resolution, 2))
        output_state_init = Value[BasicVector_[float]](output_size)
        self.state_index = self.DeclareAbstractState(output_state_init)

        # Declare Output: Updated Risk Constraint
        self.DeclareVectorOutputPort(
            "risk_constraint_output",
            output_size.shape,
            self.output_callback,
            {self.abstract_state_ticket(self.state_index)},
        )

        """ Declare Input: """
        # Full State From CrazySwarm: (x, y, z, dx, dy, dz, ddx, ddy, ddz)
        self.agent_input = self.DeclareVectorInputPort(
            "agent_position",
            9,
        ).get_index()

        # Current Adversary States: (x, y, z, dx, dy, dz)
        self.adversary_input = self.DeclareVectorInputPort(
            "obstacle_states",
            6,
        ).get_index()


        # Declare Initialization Event to Build Optimizations:
        def on_initialize(context, event):
            data = self.evaluate(context)
            self._data = data
            self.update_failure_probability()
            self.update_log_survival()
            self.update_constraints()

        self.DeclareInitializationEvent(
            event=PublishEvent(
                trigger_type=TriggerType.kInitialization,
                callback=on_initialize,
                )
            )

            # Output Update Event:
        def periodic_output_event(context, event):
            pass

        self.DeclarePeriodicEvent(
            period_sec=self._OUTPUT_UPDATE_RATE,
            offset_sec=0.0,
            event=PublishEvent(
                trigger_type=TriggerType.kPeriodic,
                callback=periodic_output_event,
                )
            )

    # Output Port Callback:
    def output_callback(self, context, output):
        pass

    # Evaluate Data:
    def evaluate(self, context) -> jnp.ndarray:
        # Get Input Port Values and Convert to Jax Array:
        agent_states = jnp.asarray(self.get_input_port(self.agent_input).Eval(context)[:3])
        adversary_states = jnp.asarray(self.get_input_port(self.adversary_input).Eval(context)[:3])

        # Evaluate if agent has failed:
        distance = jnp.sqrt(
            jnp.sum((agent_states - adversary_states) ** 2, axis=0)
        ) - self._failure_radius

        # Create Data Point Pairs: 
        if distance <= 0:
            x = distance
            y = 1.0
            self._failure_flag = True
        else:
            x = distance
            y = 0.0
            self._failure_flag = False

        return jnp.array([[x], [y]], dtype=float)

    # Record Data:
    def record_data(self, data: jax.Array) -> None:
        vector = jnp.append(self._data, data, axis=1)
        indx = jnp.argsort(vector[0, :])
        self._data = vector[:, indx]

    # Bin Values:
    def bin_data(self) -> jnp.ndarray:
        # Binning Operation
        sums, edges = jnp.histogram(
            self._data[0, :],
            bins=self._spline_resolution,
            weights=self._data[1, :],
        )
        counts, _ = jnp.histogram(
            self._data[0, :], 
            bins=self._spline_resolution,
        )
        y = np.divide(sums, counts, out=np.zeros_like(sums, dtype=float), where=counts!=0)
        x = (edges[:-1] + edges[1:]) / 2
        binned_data = jnp.vstack([x, y])

        return binned_data

    # Failure Probability Fit:
    def update_failure_probability(self, data: jax.Array):
        pass

    # Log-Survival Fit:
    def update_log_survival(self):
        pass

    # Construct Risk Constraint:
    def update_constraints(self):
        pass


