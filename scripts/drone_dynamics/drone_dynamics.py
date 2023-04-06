import numpy as np
import jax
import jax.numpy as jnp

from pydrake.common.value import Value
from pydrake.systems.framework import (
    LeafSystem,
    PublishEvent,
    TriggerType,
    BasicVector_,
)

import pdb


class DroneDynamics(LeafSystem):
    """
        Reimplementation from Crazyswarm2: https://github.com/IMRCLab/crazyswarm2
        for compatability with pydrake
    """
    def __init__(self):
        LeafSystem.__init__(self)
        # Crazyflie 2.0 Parameters:
        self.mass = 0.034  # kg
        self.J = np.array([16.571710e-6, 16.655602e-6, 29.261652e-6])  # kg m^2

        # Note: we assume here that our control is forces
        arm_length = 0.046  # m
        arm = 0.707106781 * arm_length
        t2t = 0.006  # thrust-to-torque ratio
        self.B0 = np.array([
            [1, 1, 1, 1],
            [-arm, -arm, arm, arm],
            [-arm, arm, arm, -arm],
            [-t2t, t2t, -t2t, t2t]
        ])
        self.g = 9.81  # not signed

        if self.J.shape == (3, 3):
            self.inv_J = np.linalg.pinv(self.J)  # full matrix -> pseudo inverse
        else:
            self.inv_J = 1 / self.J  # diagonal matrix -> division

        # Drake LeafSystem Parameters: Continuous States
        state_position = np.zeros((3,))
        self.position_index = self.DeclareContinuousState(
            Value[BasicVector_[float]](state_position),
        )
        state_velocity = np.zeros((3,))
        self.velocity_index = self.DeclareContinuousState(
            Value[BasicVector_[float]](state_velocity),
        )
        state_quaternion = np.zeros((4,))
        self.quaternion_index = self.DeclareContinuousState(
            Value[BasicVector_[float]](state_quaternion),
        )
        state_omega = np.zeros((3,))
        self.omega_index = self.DeclareContinuousState(
            Value[BasicVector_[float]](state_omega),
        )

    def DoCalcTimeDerivatives(self, context, derivatives):
        # Directly taken from: https://github.dev/IMRCLab/crazyswarm2/blob/main/crazyflie_sim/crazyflie_sim/backend/np.py
        def rpm_to_force(rpm):
            # polyfit using data and scripts from https://github.com/IMRCLab/crazyflie-system-id
            p = [2.55077341e-08, -4.92422570e-05, -1.51910248e-01]
            force_in_grams = np.polyval(p, rpm)
            force_in_newton = force_in_grams * 9.81 / 1000.0
            return np.maximum(force_in_newton, 0)

        force = rpm_to_force(action.rpm)

        x = (
            context.get_continuous_state_vector()
            .GetAtIndex(self.position_index)
        )
        dx = (
            context.get_continuous_state_vector()
            .GetAtIndex(self.velocity_index)
        )
        q = (
            context.get_continuous_state_vector()
            .GetAtIndex(self.quaternion_index)
        )
        dq = (
            context.get_continuous_state_vector()
            .GetAtIndex(self.omega_index)
        )
