import numpy as np

import jax.numpy as jnp

from pydrake.solvers import mathematicalprogram as mp
from pydrake.solvers.osqp import OsqpSolver
from pydrake.solvers.mathematicalprogram import SolverOptions

import failure_probability_fit as fp
import log_survival_fit as ls

import matplotlib.pyplot as plt
import pdb


def bin_data(x, y, num_bins) -> jnp.ndarray:
    # Binning Operation
    sums, edges = jnp.histogram(
        x,
        bins=num_bins,
        weights=y,
    )
    counts, _ = jnp.histogram(
        x, 
        bins=num_bins,
    )
    y = np.divide(sums, counts, out=np.zeros_like(sums, dtype=float), where=counts!=0)
    x = (edges[:-1] + edges[1:]) / 2
    binned_data = jnp.vstack([x, y])

    return binned_data


def main():
    # Make sure Bins evenly distribute across the splines:
    num_bins = 201
    num_spline = 5
    num_bins = num_bins - num_bins % num_spline

    # Create Random X and Y Data Points and bin them:
    size = 1057
    x_data = np.random.rand(size,)
    y_data = np.random.randint(2, size=size)
    binned_data = bin_data(x_data, y_data, num_bins)

    # Design Variable Vector:
    x = jnp.linspace(binned_data[0, 0], binned_data[0, -1], num_spline+1)
    y = jnp.zeros((num_spline+1,))

    # JIT functions for fp:
    H_fp_function, f_fp_function = fp.jit_functions(num_spline=num_spline)

    H_fp = H_fp_function(y, x, binned_data[1, :], binned_data[0, :])
    f_fp = f_fp_function(y, x, binned_data[1, :], binned_data[0, :])

    # Setup Optimization: (FP)
    prog = mp.MathematicalProgram()

    # Design Variables:
    opt_vars = prog.NewContinuousVariables(num_spline+1, "y")

    lb = np.zeros(opt_vars.shape)
    ub = np.ones(opt_vars.shape)

    variable_bounds = prog.AddBoundingBoxConstraint(
        lb,
        ub,
        opt_vars,
    )

    objective_function = prog.AddQuadraticCost(
        Q=H_fp,
        b=f_fp,
        vars=opt_vars,
    )

    # Solve the program:
    """OSQP:"""
    osqp = OsqpSolver()
    solver_options = SolverOptions()
    solver_options.SetOption(osqp.solver_id(), "rho", 1e-04)
    solver_options.SetOption(osqp.solver_id(), "eps_abs", 1e-05)
    solver_options.SetOption(osqp.solver_id(), "eps_rel", 1e-05)
    solver_options.SetOption(osqp.solver_id(), "eps_prim_inf", 1e-05)
    solver_options.SetOption(osqp.solver_id(), "eps_dual_inf", 1e-05)
    solver_options.SetOption(osqp.solver_id(), "max_iter", 5000)
    solver_options.SetOption(osqp.solver_id(), "polish", True)
    solver_options.SetOption(osqp.solver_id(), "polish_refine_iter", 3)
    solver_options.SetOption(osqp.solver_id(), "warm_start", True)
    solver_options.SetOption(osqp.solver_id(), "verbose", False)
    fp_solution = osqp.Solve(
        prog,
        y,
        solver_options,
    )
    fp_sol = fp_solution.GetSolution(opt_vars)

    print(f"FP Solver Status: {fp_solution.get_solver_details().status_val}")
    print(f"FP Objective Function Convex: {objective_function.evaluator().is_convex()}")

    # JIT functions for ls:
    A_ls_function, b_ls_function, H_ls_function, f_ls_function = ls.jit_functions()

    A_ls = A_ls_function(y, x)
    b_ls = -b_ls_function(y, x)

    ls_y = np.log(1 - fp_sol)
    weights = jnp.ones(ls_y.shape)
    H_ls = H_ls_function(y, ls_y, weights)
    f_ls = f_ls_function(y, ls_y, weights)

    # Setup Optimization: (LS)
    prog = mp.MathematicalProgram()

    # Design Variables:
    opt_vars = prog.NewContinuousVariables(num_spline+1, "y")

    lb = np.NINF * np.ones(opt_vars.shape)
    ub = np.zeros(opt_vars.shape)

    variable_bounds = prog.AddBoundingBoxConstraint(
        lb,
        ub,
        opt_vars,
    )

    inequality_constraints = prog.AddLinearConstraint(
        A=A_ls,
        lb=np.NINF*np.ones(b_ls.shape),
        ub=b_ls,
        vars=opt_vars,
    )

    objective_function = prog.AddQuadraticCost(
        Q=H_ls,
        b=f_ls,
        vars=opt_vars,
    )

    ls_solution = osqp.Solve(
        prog,
        y,
        solver_options,
    )
    ls_sol = ls_solution.GetSolution(opt_vars)

    print(f"LS Solver Status: {ls_solution.get_solver_details().status_val}")
    print(f"LS Objective Function Convex: {objective_function.evaluator().is_convex()}")

    # Setup Figure: FP
    fig, ax = plt.subplots()
    raw_plt, = ax.plot([], [], color='black', marker='.', linestyle='None')
    bin_plt, = ax.plot([], [], color='red', marker='.', linestyle='None')
    fit_plt, = ax.plot([], [], color='blue')
    ax.set_xlim([-1, 2])
    ax.set_ylim([-1, 2])
    ax.set_xlabel('X')  # X Label
    ax.set_ylabel('Y')  # Y Label
    ax.set_title('Test Fit:')

    # Plot the raw data:
    raw_plt.set_data(x_data, y_data)

    # Plot binned data points:
    bin_plt.set_data(binned_data[0, :], binned_data[1, :])

    # Plot the Linear Spline Fit:
    fit_plt.set_data(x, fp_sol)

    plt.show()
    plt.savefig('fp_figure.png')

    # Setup Figure: LS
    fig, ax = plt.subplots()
    data_plt, = ax.plot([], [], color='red', marker='.', linestyle='None')
    fit_plt, = ax.plot([], [], color='blue')
    ax.set_xlim([-1, 2])
    ax.set_ylim([-5, 1])
    ax.set_xlabel('X')  # X Label
    ax.set_ylabel('Y')  # Y Label
    ax.set_title('Test Fit:')

    # Plot binned data points:
    data_plt.set_data(x, ls_y)

    # Plot the Linear Spline Fit:
    fit_plt.set_data(x, ls_sol)

    plt.show()
    plt.savefig('ls_figure.png')


# Test Instantiation:
if __name__ == "__main__":
    main()
