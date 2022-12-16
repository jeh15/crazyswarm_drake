# Code Explanation:

### Intended Block Diagram of System:
![Naming Reference](/imgs/block_diagram_.png)

### High-level Code Overview:
``crazyswarm_class`` is a implementor of ``CrazySwarm`` API calls. 
Input: a full state command (position/velocity/acceleration).
Output: the current full state of the Drone.

``reference_trajectory`` provides the target position of the drone.
Input: None
Output: Target Position (x, y)

``motion_planner`` is a LeafSystem that contains a QuadraticProgram to control the drone.
Input[0] (Initial Condition): current drone state based on ``crazyswarm_class`` output.
Input[1] (Target Position): target position provided by ``reference_trajectory`` output.
Output: full state finite time horizon trajectory.

``trajectory_parser`` converts ``motion_planner`` output into ``PiecewisePolynomial.FirstOrderHold`` for interpolation.
Input: full state finite time horizon trajectory from ``motion_planner`` output.
Output: interpolated full state command based on current time.
