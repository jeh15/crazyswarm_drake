import matplotlib.pyplot as plt
import numpy as np
import time
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.primitives import LogVectorOutput, ConstantVectorSource

# DEBUG:
import pdb

# Custom LeafSystems:
import reference_trajectory

# Create a block diagram containing our system.
builder = DiagramBuilder()

# Reference Trajectory:
reference = builder.AddSystem(reference_trajectory.FigureEight())

# Logger:
logger = LogVectorOutput(reference.get_output_port(0), builder)
diagram = builder.Build()

# Set the initial conditions, x(0).
context = diagram.CreateDefaultContext()

# Create the simulator, and simulate for 1 seconds.
simulator = Simulator(diagram, context)
simulator.set_target_realtime_rate(1.0)
simulator.Initialize()

pdb.set_trace()

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