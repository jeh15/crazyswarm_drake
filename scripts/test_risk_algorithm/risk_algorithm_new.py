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

# Import Optimization Functions:
import failure_probability_regression as fp
import log_survival_regression as ls

import pdb


class RiskAlgorithm(LeafSystem):
    def __init__(self, config: ml_collections.ConfigDict()):
        LeafSystem.__init__(self)
        # Parameters:
        self._OUTPUT_UPDATE_RATE = config.motion_planner_rate
        self._spline_resolution = config.spline_resolution
        self._bin_resolution = config.bin_resolution
        self._failure_radius = config.failure_radius

        # Make sure number of bins and splines are compatable:
        if (self._bin_resolution < self._spline_resolution):
            print(f"Defaulting bin_resolution == spline_resolution... \nSuggesting that bin_resolution >> spline_resolution.")
            self._bin_resolution = self._spline_resolution
        self._bin_resolution = self._bin_resolution - self._bin_resolution % self._spline_resolution
        if (self._bin_resolution != config.bin_resolution):
            print(f"Changed bin_resolution to {self._bin_resolution} to be compatable with spline_resolution.")

        # Create instances of namespace containers:
        self.fpn = fp.FailureProbabilityNamespace(
            spline_resolution=self._spline_resolution,
        )
        self.lsn = ls.LogSurvivalNamespace(
            spline_resolution=self._spline_resolution,
        )

        # Initialize data vector:
        self.data = []

        # Initialize Abstract States: (Output Only)
        output_size = np.zeros((self._spline_resolution, 2)).flatten()
        output_state_init = Value[BasicVector_[float]](output_size)
        self.state_index = self.DeclareAbstractState(output_state_init)

        # Declare Output: Updated Risk Constraint
        self.DeclareVectorOutputPort(
            "risk_constraint_output",
            output_size.shape[0],
            self.output_callback,
            {self.abstract_state_ticket(self.state_index)},
        )

        # Declare Input:
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
            self.fpn.configure_solver()
            self.lsn.configure_solver()
            initial_data = self.evaluate(context)
            self.data = initial_data
            data = self.bin_data()
            fp_solution = self.fpn.initialize_optimization(data=data)
            ls_solution = self.lsn.initialize_optimization(data=fp_solution)
            self.constraints = self.update_constraints(data=ls_solution)
            a_state = context.get_mutable_abstract_state(self.state_index)
            a_state.set_value(self.constraints.flatten())

        self.DeclareInitializationEvent(
            event=PublishEvent(
                trigger_type=TriggerType.kInitialization,
                callback=on_initialize,
                )
            )

        #TODO(jeh15): Evaluate and calculate data faster than resolving QPs

        # Output Update Event:
        def periodic_output_event(context, event):
            current_data = self.evaluate(context)
            self.record_data(current_data)
            data = self.bin_data()
            fp_solution = self.fpn.update(data=data)
            ls_solution = self.lsn.update(data=fp_solution)
            self.constraints = self.update_constraints(data=ls_solution)
            a_state = context.get_mutable_abstract_state(self.state_index)
            a_state.set_value(self.constraints.flatten())

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
        a_state = context.get_mutable_abstract_state(self.state_index)
        a_value = a_state.get_mutable_value()
        output.SetFromVector(a_value.get_mutable_value())

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
        vector = jnp.append(self.data, data, axis=1)
        indx = jnp.argsort(vector[0, :])
        self.data = vector[:, indx]

    # Bin Values:
    def bin_data(self) -> jnp.ndarray:
        # Binning Operation
        sums, edges = jnp.histogram(
            self.data[0, :],
            bins=self._bin_resolution,
            weights=self.data[1, :],
        )
        counts, _ = jnp.histogram(
            self.data[0, :],
            bins=self._bin_resolution,
        )
        y = np.divide(
            sums,
            counts,
            out=np.zeros_like(sums, dtype=float),
            where=counts!=0,
        )
        x = (edges[:-1] + edges[1:]) / 2
        binned_data = jnp.vstack([x, y])

        return binned_data

    # Construct Risk Constraint:
    def update_constraints(self, data: jax.Array) -> jnp.ndarray:
        poly_coefficients = []
        for i in range(0, self._spline_resolution):
            p = np.polyfit(data[0, i:i+2], data[1, i:i+2], 1)
            poly_coefficients.append(p)
        return jnp.asarray(poly_coefficients)
