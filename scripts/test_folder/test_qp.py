from pydrake.solvers import MathematicalProgram, Solve
import numpy as np

prog = MathematicalProgram()
x = prog.NewContinuousVariables(3, "x")
prog.AddQuadraticCost(x[0] * x[0] + 2 * x[0] + 3)
# Adds the quadratic cost on the squared norm of the vector
# (x[1] + 3*x[2] - 1, 2*x[1] + 4*x[2] -4)
prog.Add2NormSquaredCost(A = [[1, 3], [2, 4]], b=[1, 4], vars=[x[1], x[2]])

# Adds the linear constraints.
prog.AddLinearEqualityConstraint(x[0] + 2*x[1] == 5)
prog.AddLinearConstraint(x[0] + 4 *x[1] <= 10)
# Sets the bounds for each variable to be within [-1, 10]
prog.AddBoundingBoxConstraint(-1, 10, x)

# Solve the program.
result = Solve(prog)
print(f"optimal solution x: {result.GetSolution(x)}")
print(f"optimal cost: {result.get_optimal_cost()}")