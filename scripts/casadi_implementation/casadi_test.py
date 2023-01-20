import numpy as np
import casadi as ca

import pdb


def equality_constraints(states, initial_conditions, friction, mass, dt):
    """
    Helper Functions:
        1. Euler Collocation
    """

    # Euler Collocation:
    def collocation_constraint(x, dt):
        return ca.horzcat(x[0][1:] - x[0][:-1] - x[1][:-1] * dt)

    """
    Equality Constraints:
        1. Initial Condition
        2. Collocation Constraint
    """

    # Create Handles:
    x = states[0, :]
    y = states[1, :]
    dx = states[2, :]
    dy = states[3, :]
    ux = states[4, :]
    uy = states[5, :]

    # 1. Initial Condition Constraints:
    # ic_constraint = states[:, 0] - initial_conditions[:]

    ic_constraint = ca.horzcat(
        x[0] - initial_conditions[0],
        y[0] - initial_conditions[1],
        dx[0] - initial_conditions[2],
        dy[0] - initial_conditions[3],
        ux[0] - initial_conditions[4],
        uy[0] - initial_conditions[5],
    )

    # 2. Collocation Constraints:
    ddx = (ux - friction * dx) / mass
    ddy = (uy - friction * dy) / mass
    x_defect = collocation_constraint([x, dx], dt)
    y_defect = collocation_constraint([y, dy], dt)
    dx_defect = collocation_constraint([dx, ddx], dt)
    dy_defect = collocation_constraint([dy, ddy], dt)

    equality_constraint = ca.horzcat(
        ic_constraint,
        x_defect,
        y_defect,
        dx_defect,
        dy_defect,
    )

    return equality_constraint

def main():
    # Test CasADi Codegen:

    # Model Parameters:
    time_horizon = 1.0
    friction = 0.01
    mass = 1.0

    # Optimization Parameters:
    num_states = 6
    num_nodes = 21
    dt = time_horizon / (num_nodes - 1)

    # Initialize Optimization Variables:
    x = ca.SX.sym('x', num_states, num_nodes)
    ic = ca.SX.sym('ic', num_states, 1)

    # Calculate Symbolic Equality Constraints:
    constraints = equality_constraints(x, ic, friction, mass, dt)

    # Calculate A and b matrices:
    A_eq = ca.jacobian(constraints, x)
    b_eq = -equality_constraints(ca.SX.zeros(x.shape), ic, friction, mass, dt)


if __name__ == "__main__":
    main()
