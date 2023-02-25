import numpy as np
import pdb

# Get Values for Halfspace-Constraint:
previous_trajectory = np.ones((6, 21), dtype=float)
obstacle_states = np.ones((6,), dtype=float)
time_vector = np.linspace(0, 1, 21)

# Linear prediction model of adversary:
obstacle_trajectory = np.einsum('i,j->ij', obstacle_states[3:5], time_vector) \
    + obstacle_states[:2].reshape((2, -1))
# Halfspace vectors to linearize about:
halfspace_position_vector = obstacle_trajectory - previous_trajectory[:2, :]
halfspace_velocity_vector = obstacle_states[3:5].reshape((2, -1)) - previous_trajectory[3:5, :]

# Default if halfspace vector is a null vector:
halfspace_position_magnitude = np.linalg.norm(halfspace_position_vector)
halfspace_position_squared = np.einsum('ij,ij->j', halfspace_position_vector, halfspace_position_vector)

halfspace_position_ratio = np.divide(
    halfspace_position_magnitude,
    halfspace_position_squared,
    out=np.zeros_like(halfspace_position_squared),
    where=halfspace_position_squared!=0.0,
)

halfspace_velocity_magnitude = np.linalg.norm(halfspace_velocity_vector)
halfspace_velocity_squared = np.einsum('ij,ij->j', halfspace_velocity_vector, halfspace_velocity_vector)
halfspace_velocity_ratio = np.divide(
    halfspace_velocity_magnitude,
    halfspace_velocity_squared,
    out=np.zeros_like(halfspace_velocity_squared),
    where=halfspace_velocity_squared!=0,
)

pdb.set_trace()