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
        self._UPDATE_RATE = config.sample_rate
        self._OUTPUT_UPDATE_RATE = config.motion_planner_rate
        self._spline_resolution = config.spline_resolution
        self._bin_resolution = config.bin_resolution
        self._failure_radius = config.failure_radius
        self._basis_vector_dim = config.candidate_sources_dimension

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
        constraint_output_size = np.zeros((self._spline_resolution, 2)).flatten()
        constraint_output_state_init = Value[BasicVector_[float]](constraint_output_size)
        self.constraint_state_index = self.DeclareAbstractState(constraint_output_state_init)

        basis_vector_output_size = np.zeros((self._basis_vector_dim,)).flatten()
        basis_vector_output_state_init = Value[BasicVector_[float]](basis_vector_output_size)
        self.basis_vector_state_index = self.DeclareAbstractState(basis_vector_output_state_init)

        # Declare Output: Updated Risk Constraint
        self.constraint_output = self.DeclareVectorOutputPort(
            "risk_constraint_output",
            constraint_output_size.shape[0],
            self.output_constraint_callback,
            {self.abstract_state_ticket(self.constraint_state_index)},
        ).get_index()

        # Declare Output: Updated Risk Constraint
        self.basis_vector_output = self.DeclareVectorOutputPort(
            "basis_vector_output",
            basis_vector_output_size.shape[0],
            self.output_basis_vector_callback,
            {self.abstract_state_ticket(self.basis_vector_state_index)},
        ).get_index()

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
            projected_data, self.basis_vector = self.project_data(self.data)
            data = self.bin_data(projected_data)
            fp_solution = self.fpn.initialize_optimization(data=data)
            log_transform = np.vstack([fp_solution[0, :], np.log(1-fp_solution[1, :])])
            ls_solution = self.lsn.initialize_optimization(data=log_transform)
            self.constraints = self.update_constraints(data=ls_solution)
            constraint_state = context.get_mutable_abstract_state(self.constraint_state_index)
            constraint_state.set_value(self.constraints.flatten())
            basis_vector_state = context.get_mutable_abstract_state(self.basis_vector_state_index)
            basis_vector_state.set_value(self.basis_vector.flatten())
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
            projected_data, self.basis_vector = self.project_data(self.data)
            data = self.bin_data(projected_data)
            fp_solution = self.fpn.initialize_optimization(data=data)
            log_transform = np.vstack([fp_solution[0, :], np.log(1-fp_solution[1, :])])
            ls_solution = self.lsn.initialize_optimization(data=log_transform)
            self.constraints = self.update_constraints(data=ls_solution)
            constraint_state = context.get_mutable_abstract_state(self.constraint_state_index)
            constraint_state.set_value(self.constraints.flatten())
            basis_vector_state = context.get_mutable_abstract_state(self.basis_vector_state_index)
            basis_vector_state.set_value(self.basis_vector.flatten())
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
    def output_constraint_callback(self, context, constraint_output):
        constraint_state = context.get_mutable_abstract_state(self.constraint_state_index)
        constraint_value = constraint_state.get_mutable_value()
        constraint_output.SetFromVector(constraint_value.get_mutable_value())

    def output_basis_vector_callback(self, context, basis_vector_output):
        basis_vector_state = context.get_mutable_abstract_state(self.basis_vector_state_index)
        basis_vector_value = basis_vector_state.get_mutable_value()
        basis_vector_output.SetFromVector(basis_vector_value.get_mutable_value())

    # Evaluate Data:
    def evaluate(self, context) -> np.ndarray:
        # Get Input Port Values and Convert to Numpy Array:
        agent_states = np.asarray(self.get_input_port(self.agent_input).Eval(context))
        adversary_states = np.asarray(self.get_input_port(self.adversary_input).Eval(context))

        # 2D Euclidean Distance:
        distance = np.linalg.norm((agent_states[:2] - adversary_states[:2]))
        relative_velocity = np.linalg.norm((agent_states[3:5] - adversary_states[3:5]))
        eval_distance = distance - self._failure_radius

        # Create Data Point Pairs:
        if eval_distance <= 0:
            x = distance
            y = relative_velocity
            label = 1.0
            self._failure_flag = True
        else:
            x = distance
            y = relative_velocity
            label = 0.0
            self._failure_flag = False
        
        return np.array([[x], [y], [label]], dtype=float)

    # Record Data:
    def record_data(self, data: np.ndarray) -> None:
        vector = np.append(self.data, data, axis=1)
        indx = np.argsort(vector[0, :])
        self.data = vector[:, indx]

    # Bin Values:
    def bin_data(self, data: np.ndarray) -> np.ndarray:
        # Binning Operation
        sums, edges = np.histogram(
            data[0, :],
            bins=self._bin_resolution,
            weights=data[1, :],
        )
        counts, _ = np.histogram(
            data[0, :],
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

    def project_data(self, data: np.ndarray):
        # Create mask for labeled data:
        success_mask = data[-1, :] == 0
        data_success = data[:-1, success_mask]
        data_failure = data[:-1, ~success_mask]
        # Find centroids:
        centroid_success = np.mean(data_success, axis=1)
        centroid_failure = np.mean(data_failure, axis=1)
        vector =  centroid_success - centroid_failure
        if np.isnan(vector).any():
            vector = np.zeros_like(vector)
        magnitude = np.linalg.norm(vector)
        # Find basis vector from centroids:
        basis_vector = np.divide(vector, magnitude, out=np.zeros_like(vector), where=magnitude!=0)
        # Project data onto unit basis vector:
        data_projected = np.einsum('ij,i->j', data[:-1, :], basis_vector)
        # Transform data back to original axes: (For Debug/Logging)
        self._data_projected_origin = data_projected * np.reshape(basis_vector, (-1, 1))
        data_projected = np.vstack([data_projected, data[-1, :]])
        return data_projected, basis_vector