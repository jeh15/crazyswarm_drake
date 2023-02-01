import numpy as np

import jax
import jax.numpy as jnp
from jax import (
    jit,
    jacfwd,
    jacrev
)

from pydrake.solvers import mathematicalprogram as mp
from pydrake.solvers.osqp import OsqpSolver
from pydrake.solvers.mathematicalprogram import SolverOptions

import failure_probability_fit as fp

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
    num_bins = 7
    num_spline = 2
    num_bins = num_bins - num_bins % num_spline

    # Create Random X and Y Data Points and bin them:
    size = 957
    x_data = np.random.rand(size,)
    y_data = np.random.randint(2, size=size)
    binned_data = bin_data(x_data, y_data, num_bins)

    y = jnp.zeros((num_spline+1,))

    H_function, f_function = fp.jit_functions(num_spline=num_spline)

    H = H_function(y, binned_data[0, :], binned_data[1, :])
    f = f_function(y, binned_data[0, :], binned_data[1, :])

    # Setup Optimization:
    # Construct OSQP Problem:
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
        Q=H,
        b=f,
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
    solution = osqp.Solve(
        prog,
        y,
        solver_options,
    )

    print(f"Solver Status: {solution.get_solver_details().status_val}")
    print(f"Objective Function Convex: {objective_function.evaluator().is_convex()}")

    pdb.set_trace()

    # Setup Figure: Initialize Figure / Axe Handles
    fig, ax = plt.subplots()
    fig.patch.set_alpha(0.0)
    raw_plt, = ax.plot([], [], color='black', marker='.', linestyle = 'None')
    bin_plt, = ax.plot([], [], color='red', marker='.', linestyle = 'None')
    fit_plt, = ax.plot([], [], color='blue')
    ax.axis('equal')
    ax.set_xlabel('X')  # X Label
    ax.set_ylabel('Y')  # Y Label
    ax.set_title('Test Fit:')

    # Plot the raw data:
    raw_plt.set_data(x_data, y_data)

    # Plot binned data points:
    bin_plt.set_data(binned_data[0, :], binned_data[1, :])

    # Plot the Linear Spline Fit:
    x_fit = jnp.linspace(binned_data[0, 0], binned_data[0, -1], num_spline + 1)
    fit_plt.set_data(x_fit, solution.GetSolution(opt_vars))
    
    plt.show()


# Test Instantiation:
if __name__ == "__main__":
    main()