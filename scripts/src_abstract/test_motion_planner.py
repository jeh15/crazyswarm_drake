import matplotlib.pyplot as plt
import numpy as np
import time
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.primitives import LogVectorOutput, ConstantVectorSource

# DEBUG:
import pdb

# Custom LeafSystems:
import motion_planner
import reference_trajectory
import trajectory_parser
import crazyswarm_class

# Create a block diagram containing our system.
builder = DiagramBuilder()

# Reference Trajectory:
reference = builder.AddSystem(reference_trajectory.FigureEight())

# Trajectory Parser:
parser = builder.AddSystem(trajectory_parser.TrajectoryParser())

# Motion Planner:
driver_planner = motion_planner.QuadraticProgram()
planner = builder.AddSystem(driver_planner)

# CrazySwarm API:
driver_system = crazyswarm_class.CrazyswarmSystem()
system = builder.AddSystem(driver_system)

# Connect Reference Trajectory to Motion Planner:
builder.Connect(reference.get_output_port(0), planner.get_input_port(driver_planner.target_input))

# Connect Motion Planner to Trajectory Parser:
builder.Connect(planner.get_output_port(0), parser.get_input_port(0))

# Connect Trajectory Parser to CrazySwarm:
builder.Connect(parser.get_output_port(0), system.get_input_port(0))

# Connect Drone Output to Motion Planner:
dummy = builder.AddSystem(ConstantVectorSource(np.zeros((9,), dtype=float)))
builder.Connect(system.get_output_port(0), planner.get_input_port(driver_planner.initial_condition_input))

# Logger:
logger = LogVectorOutput(system.get_output_port(0), builder)
diagram = builder.Build()

# Set the initial conditions, x(0).
context = diagram.CreateDefaultContext()

# Create the simulator, and simulate for 1 seconds.
simulator = Simulator(diagram, context)
simulator.set_target_realtime_rate(1.0)
simulator.Initialize()

# Simulate System:
FINAL_TIME = 3.0
dt = 1.0 / 100.0
next_time_step = dt

_start = time.perf_counter()
while next_time_step <= FINAL_TIME:
    print(f"Drake Real Time Rate: {simulator.get_actual_realtime_rate()}") 
    simulator.AdvanceTo(next_time_step)
    next_time_step += dt

_end = time.perf_counter() - _start
print(f"Time: {_end}")

# Plot the results:
log = logger.FindLog(context)
plt.figure()
plt.plot(log.sample_times(), log.data().transpose())
pdb.set_trace()
plt.xlabel('t')
plt.ylabel('x(t)')
plt.show()