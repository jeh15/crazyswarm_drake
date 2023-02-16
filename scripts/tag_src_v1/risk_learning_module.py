from functools import partial

import numpy as np
import ml_collections

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


class RiskLearning(LeafSystem):
    def __init__(self, config: ml_collections.ConfigDict()):
        LeafSystem.__init__(self)
        # Parameters:
        self._UPDATE_RATE = config.crazyswarm_rate
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
            log_transform = np.vstack([fp_solution[0, :], np.log(1-fp_solution[1, :])])
            ls_solution = self.lsn.initialize_optimization(data=log_transform)
            self.constraints = self.update_constraints(data=ls_solution)
            a_state = context.get_mutable_abstract_state(self.state_index)
            a_state.set_value(self.constraints.flatten())
            # Visibility for Debug:
            self._fp_sol = fp_solution
            self._ls_sol = ls_solution
            self._data = data

        self.DeclareInitializationEvent(
            event=PublishEvent(
                trigger_type=TriggerType.kInitialization,
                callback=on_initialize,
                )
            )

        # Record Data at Control Rate:
        def periodic_update_event(context, event):
            current_data = self.evaluate(context)
            self.record_data(current_data)

        self.DeclarePeriodicEvent(
            period_sec=self._UPDATE_RATE,
            offset_sec=0.0,
            event=PublishEvent(
                trigger_type=TriggerType.kPeriodic,
                callback=periodic_update_event,
            )
        )

        # Output Update Event:
        def periodic_output_event(context, event):
            current_data = self.evaluate(context)
            self.record_data(current_data)
            data = self.bin_data()
            fp_solution = self.fpn.update(data=data)
            log_transform = np.vstack([fp_solution[0, :], np.log(1-fp_solution[1, :])])
            ls_solution = self.lsn.initialize_optimization(data=log_transform)
            self.constraints = self.update_constraints(data=ls_solution)
            a_state = context.get_mutable_abstract_state(self.state_index)
            a_state.set_value(self.constraints.flatten())
            # Visibility for Debug/Logging:
            self._fp_sol = fp_solution
            self._ls_sol = ls_solution
            self._data = data

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
    def evaluate(self, context) -> np.ndarray:
        # Get Input Port Values and Convert to Numpy Array:
        agent_states = np.asarray(self.get_input_port(self.agent_input).Eval(context)[:2])
        adversary_states = np.asarray(self.get_input_port(self.adversary_input).Eval(context)[:2])

        # 3D Euclidean Distance:
        # agent_states = np.asarray(self.get_input_port(self.agent_input).Eval(context)[:3])
        # adversary_states = np.asarray(self.get_input_port(self.adversary_input).Eval(context)[:3])
        # distance = np.sqrt(
        #     np.sum((agent_states - adversary_states) ** 2, axis=0)
        # )

        # 2D Euclidean Distance:
        distance = np.linalg.norm((agent_states - adversary_states))
        eval_distance = distance - self._failure_radius

        # Create Data Point Pairs:
        if eval_distance <= 0:
            x = distance
            y = 1.0
            self._failure_flag = True
        else:
            x = distance
            y = 0.0
            self._failure_flag = False

        return np.array([[x], [y]], dtype=float)

    # Record Data:
    def record_data(self, data: np.ndarray) -> None:
        vector = np.append(self.data, data, axis=1)
        indx = np.argsort(vector[0, :])
        self.data = vector[:, indx]

    # Bin Values:
    def bin_data(self) -> np.ndarray:
        # Binning Operation
        sums, edges = np.histogram(
            self.data[0, :],
            bins=self._bin_resolution,
            weights=self.data[1, :],
        )
        counts, _ = np.histogram(
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
        binned_data = np.vstack([x, y])

        return binned_data

    # Construct Risk Constraint:
    def update_constraints(self, data: np.ndarray) -> np.ndarray:
        poly_coefficients = []
        for i in range(0, self._spline_resolution):
            p = np.polyfit(data[0, i:i+2], data[1, i:i+2], 1)
            poly_coefficients.append(p)
        return np.asarray(poly_coefficients)
