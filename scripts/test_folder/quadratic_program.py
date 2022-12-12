from pydrake.solvers import MathematicalProgram, Solve
import numpy as np
import time
import pdb
def main():
    # Initialize MathematicalProgram:
    prog = MathematicalProgram()

    # Program Parameters:
    dim_size = 2
    state_size = dim_size * 2
    full_size = dim_size * 3
    num_nodes = 101
    time_horizon = 1
    dt = time_horizon / (num_nodes - 1)

    # Model Parameters:
    mass = 0.486
    friction = 0.1 

    # State and Control Inpute Variables:
    x   = prog.NewContinuousVariables(num_nodes, "x")
    y   = prog.NewContinuousVariables(num_nodes, "y")
    dx  = prog.NewContinuousVariables(num_nodes, "dx")
    dy  = prog.NewContinuousVariables(num_nodes, "dy")
    ux  = prog.NewContinuousVariables(num_nodes, "ux")
    uy  = prog.NewContinuousVariables(num_nodes, "uy")

    # Create Convenient Arrays:
    q = np.vstack(np.array([x, y, dx, dy, ux, uy]))

    _test_var = np.vstack(np.array([x, dx])).flatten()

    # Initial Condition Constraints:
    bounds = np.zeros(full_size, dtype=float)
    _A_initial_condition = np.eye(full_size, dtype=float)
    constraint_initial_condition = prog.AddLinearConstraint(
        A=_A_initial_condition,
        lb=bounds,
        ub=bounds,
        vars=q[:, 0]
    )

    # Add Lower and Upper Bounds: (Slower)
    # lb_bounds = np.einsum('i,j -> ij', np.array([-5, -5, -5, -5, -1, -1]), np.ones(num_nodes, dtype=float)).flatten()
    # ub_bounds = np.einsum('i,j -> ij', np.array([5, 5, 5, 5, 1, 1]), np.ones(num_nodes, dtype=float)).flatten()
    # _A_state_bounds = np.eye(full_size * num_nodes, dtype=float)
    # constraint_state_bounds = prog.AddLinearConstraint(
    #     A=_A_state_bounds,
    #     lb=lb_bounds,
    #     ub=ub_bounds,
    #     vars=q.flatten()
    # )

    # Add Lower and Upper Bounds: (Fastest)
    prog.AddBoundingBoxConstraint(-5, 5, x)
    prog.AddBoundingBoxConstraint(-5, 5, y)
    prog.AddBoundingBoxConstraint(-5, 5, dx)
    prog.AddBoundingBoxConstraint(-5, 5, dy)
    prog.AddBoundingBoxConstraint(-1, 1, ux)
    prog.AddBoundingBoxConstraint(-1, 1, uy)


    # # Collocation Constraints: (Separated)
    # bounds = np.zeros(2, dtype=float)
    # _A_collocation = np.array([[1, -1, dt, 0, 0], [0, 0, 1 - friction / mass * dt, -1, dt / mass]], dtype=float)
    # for i in range(num_nodes - 1):
    #     prog.AddLinearConstraint(
    #         A=_A_collocation,
    #         lb=bounds,
    #         ub=bounds,
    #         vars=np.array([x[i], x[i+1], dx[i], dx[i+1], ux[i]])
    #     )
    #     prog.AddLinearConstraint(
    #         A=_A_collocation,
    #         lb=bounds,
    #         ub=bounds,
    #         vars=np.array([y[i], y[i+1], dy[i], dy[i+1], uy[i]])
    #     )

    # # Collocation Constraints: (Combined)
    # bounds = np.zeros(4, dtype=float)
    # _A_collocation = np.array([
    #                         [1, -1, dt, 0, 0, 0, 0, 0, 0, 0], 
    #                         [0, 0, 1 - friction / mass * dt, -1, dt / mass, 0, 0, 0, 0, 0],
    #                         [0, 0, 0, 0, 0, 1, -1, dt, 0, 0],
    #                         [0, 0, 0, 0, 0, 0, 0, 1 - friction / mass * dt, -1, dt / mass]
    #                         ], dtype=float)
    # for i in range(num_nodes - 1):
    #     prog.AddLinearConstraint(
    #         A=_A_collocation,
    #         lb=bounds,
    #         ub=bounds,
    #         vars=np.array([x[i], x[i+1], dx[i], dx[i+1], ux[i], y[i], y[i+1], dy[i], dy[i+1], uy[i]])
    #     )

    # Collocation Constraints: Python Function
    def collocation_func(z):
        _defect = z[0][1:] - z[0][:-1] - z[1][:-1] * dt
        return _defect

    ddx, ddy = (ux - friction * dx) / mass, (uy - friction * dy) / mass
    _x_defect   = collocation_func([x,  dx])
    _dx_defect  = collocation_func([dx, ddx])
    _y_defect   = collocation_func([y,  dy])
    _dy_defect  = collocation_func([dy, ddy])
    _expr_array = np.asarray([_x_defect, _dx_defect, _y_defect, _dy_defect]).flatten()

    bounds = np.zeros(4 * (num_nodes-1), dtype=float)
    defect_constraint = prog.AddLinearConstraint(
        _expr_array, 
        lb=bounds, 
        ub=bounds 
        )

    # Collocation Constraint: np.vectorize
    # def collocation_func(z0, z1, dz):
    #     _defect = z1 - z0 - dz * dt
    #     return _defect

    # vfunc = np.vectorize(collocation_func)

    # ddx, ddy = (ux - friction * dx) / mass, (uy - friction * dy) / mass
    # _x_defect   = vfunc(x[:-1], x[1:], dx[:-1])
    # _dx_defect  = vfunc(dx[:-1], dx[1:], ddx[:-1])
    # _y_defect   = vfunc(y[:-1], y[1:], dy[:-1])
    # _dy_defect  = vfunc(dy[:-1], dy[1:], ddy[:-1])
    # _expr_array = np.hstack([_x_defect, _dx_defect, _y_defect, _dy_defect])

    # bounds = np.zeros(4 * (num_nodes-1), dtype=float)
    # defect_constraint = prog.AddLinearConstraint(
    #     _expr_array, 
    #     lb=bounds, 
    #     ub=bounds 
    #     )

    # Collocation Constraints: (Linear Constraints via forloop)
    # for i in range(num_nodes - 1):
    #     _ux, _uy = (ux[i] - friction * dx[i]) / mass, (uy[i] - friction * dy[i]) / mass
    #     prog.AddLinearConstraint(x[i] + dx[i] * dt == x[i+1])
    #     prog.AddLinearConstraint(dx[i] + _ux*dt == dx[i+1])
    #     prog.AddLinearConstraint(y[i] + dy[i] * dt == y[i+1])
    #     prog.AddLinearConstraint(dy[i] + _uy*dt == dy[i+1])

    # Objective Function:
    x_d, y_d = 1, 1
    _target = np.array([[x_d],[y_d]], dtype=float)
    _error = _target - q[:2, :]
    objective_task = prog.AddQuadraticCost(np.sum(_error ** 2))

    # Solve the program.
    result = Solve(prog)
    print(f"optimal solution x: {result.GetSolution(x)}")
    print(f"optimal solution y: {result.GetSolution(y)}")
    print(f"optimal cost: {result.get_optimal_cost()}")

if __name__ == "__main__":
    _start_time = time.perf_counter()
    main()
    print(f"Total Elaplsed Time: {time.perf_counter() - _start_time}")