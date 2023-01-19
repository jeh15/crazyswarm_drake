import numpy as np
from pydrake.solvers import mathematicalprogram as mp

def main():
    # Test Case LinearConstraint Binding:
    A = np.array([[1, 3, 4], [2., 4., 5]])
    lb = np.array([1, 2.])
    ub = np.array([3., 4.])

    prog = mp.MathematicalProgram()
    x = prog.NewContinuousVariables(3, "x")

    linear_constraint = prog.AddLinearConstraint(
        A=A,
        lb=lb,
        ub=ub,
        vars=x,
    )

    linear_constraint.evaluator().UpdateCoefficients(
        new_A=np.array([[1E-10, 0, 0], [0, 1, 1]]),
        new_lb=np.array([2, 3]),
        new_ub=np.array([3, 4]),
    )

    linear_constraint.evaluator().RemoveTinyCoefficient(tol=1E-5)


if __name__ == "__main__":
    main()