import matplotlib.pyplot as plt
import numpy as np

from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.primitives import LogVectorOutput, ConstantVectorSource

import pdb

# Custom LeafSystems:
import motion_planner

# Create a block diagram containing our system.
builder = DiagramBuilder()

# Motion Planner:
driver_planner = motion_planner.QuadraticProgram()
planner = builder.AddSystem(driver_planner)

# Create Dummy Inputs:
dummy_target_position = builder.AddSystem(ConstantVectorSource(np.ones((2,), dtype=float)))
driver_ic = ConstantVectorSource(np.zeros((6,), dtype=float))
dummy_initial_condition = builder.AddSystem(driver_ic)

# Connect Systems:
builder.Connect(
    dummy_target_position.get_output_port(0),
    planner.get_input_port(driver_planner.target_input)
    )

builder.Connect(
    dummy_initial_condition.get_output_port(0),
    planner.get_input_port(driver_planner.initial_condition_input)
    )

# Logger:
logger = LogVectorOutput(planner.get_output_port(0), builder)
diagram = builder.Build()

# Set the initial conditions, x(0).
context = diagram.CreateDefaultContext()

# Create the simulator:
simulator = Simulator(diagram, context)
simulator.set_target_realtime_rate(1.0)
simulator.Initialize()

# Simulate System:
FINAL_TIME = 5.0
dt = 1.0 / 100.0
next_time_step = dt

while next_time_step <= FINAL_TIME:
    print(f"Drake Real Time Rate: {simulator.get_actual_realtime_rate()}")
    simulator.AdvanceTo(next_time_step)
    # Get Subsystem Context for ConstantSourceVector:
    subsystem_context = driver_ic.GetMyContextFromRoot(context)
    src_value = driver_ic.get_mutable_source_value(subsystem_context)
    # Get end states to set as next IC:
    new_ic = np.reshape(
        driver_planner._full_state_trajectory,
        (-1, driver_planner._num_nodes)
        )[:, -1]
    src_value.set_value(new_ic)
    # Increment time step:
    next_time_step += dt

# Plot the results:
log = logger.FindLog(context)
plt.figure()
plt.plot(log.data()[0, :], log.data()[1, :])
plt.xlabel('x(t)')
plt.ylabel('y(t)')
plt.show()
plt.savefig('foo.png')
