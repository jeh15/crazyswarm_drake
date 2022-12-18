import matplotlib.pyplot as plt

from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.primitives import LogVectorOutput

# DEBUG:
import pdb

# Custom LeafSystems:
import reference_trajectory

# Create a block diagram containing our system.
builder = DiagramBuilder()

# Reference Trajectory:
driver_reference = reference_trajectory.FigureEight()
reference = builder.AddSystem(driver_reference)

# Logger: (Causing Output Erro)
logger = LogVectorOutput(reference.get_output_port(0), builder)
diagram = builder.Build()

# Set the initial conditions, x(0).
context = diagram.CreateDefaultContext()

# Create the simulator:
simulator = Simulator(diagram, context)
simulator.set_target_realtime_rate(1.0)
simulator.Initialize()

# Simulate System:
FINAL_TIME = 3.0
dt = 1.0 / 100.0
next_time_step = dt

while next_time_step <= FINAL_TIME:
    print(f"Drake Real Time Rate: {simulator.get_actual_realtime_rate()}")
    simulator.AdvanceTo(next_time_step)
    next_time_step += dt

# Plot the results:
log = logger.FindLog(context)
plt.figure()
plt.plot(log.data()[0, :], log.data()[1, :])
plt.xlabel('x(t)')
plt.ylabel('y(t)')
plt.show()
plt.savefig('foo.png')
