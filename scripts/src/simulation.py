import matplotlib.pyplot as plt
import time
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.primitives import LogVectorOutput, ConstantVectorSource

# Custom LeafSystems:
import crazyswarm_class
import drake_figure8

# Create a simple block diagram containing our system.
builder = DiagramBuilder()
# Custom Reference Trajectory LeafSystem:
reference_trajectory = builder.AddSystem(drake_figure8.Trajectory_Figure8())
# CrazySwarm API LeafSystem:
driver = crazyswarm_class.CrazyswarmSystem()
system = builder.AddSystem(driver)
# Connect Reference Trajectory to CrazySwarm:
builder.Connect(reference_trajectory.get_output_port(0), system.get_input_port(0))
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
FINAL_TIME = 10.0
dt = 1.0 / 100.0
next_time_step = dt

_start = time.perf_counter()
while next_time_step <= FINAL_TIME:
    try:
        print(f"Drake Real Time Rate: {simulator.get_actual_realtime_rate()}") 
        simulator.AdvanceTo(next_time_step)
        next_time_step += dt
    except:
        driver.execute_landing_sequence()

# Call End Event:
driver.execute_landing_sequence()

_end = time.perf_counter() - _start
print(f"Time: {_end}")

# Plot the results.
log = logger.FindLog(context)
plt.figure()
plt.plot(log.sample_times(), log.data().transpose())
plt.xlabel('t')
plt.ylabel('x(t)')
plt.show()