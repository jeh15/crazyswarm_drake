import numpy as np
import pydrake.systems.scalar_conversion

from pydrake.systems.framework import LeafSystem_, SystemScalarConverter, PortDataType
from pydrake.systems.scalar_conversion import TemplateSystem
from pydrake.autodiffutils import AutoDiffXd

@TemplateSystem.define("DroneSystem_")
def DroneSystem_(T):
    """
    2D Double Integrator System:
    """

    class Impl(LeafSystem_[T]):

        def _construct(self, converter=None):
            LeafSystem_[T].__init__(self, converter=converter)
            # Declare Input: [acceleration/force]
            self.DeclareVectorInputPort("u", 2)
            # Declate System States: [position, velocity]
            state_index = self.DeclareContinuousState(2, 2, 0)
            # Declare Output: Full-State
            self.DeclareStateOutputPort("x", state_index)

            # System Parameters:
            self.mass = 0.486   # Mass of Drone
            self.friction = 0.1 # Damping Coefficient

        def _construct_copy(self, other, converter=None):
            Impl._construct(self, converter=converter)

        def DoCalcTimeDerivatives(self, context, derivatives):
            x = context.get_continuous_state_vector().CopyToVector()
            u = self.EvalVectorInput(context, 0).CopyToVector()
            q = x[:2]
            dq = x[2:]
            ddq = np.array([(u[0] - self.friction * dq[0]) / self.mass,
                            (u[1] - self.friction * dq[1]) / self.mass])
            derivatives.get_mutable_vector().SetFromVector(np.concatenate((dq, ddq)))

    return Impl

DroneSystem = DroneSystem_[None]  # Default instantiation

# Main Function:
def main():
    DroneSystem_float = DroneSystem_[float]
    DroneSystem_autodiff = DroneSystem_float().ToAutoDiffXd()
    print(DroneSystem_float)
    print(DroneSystem_autodiff)


# Test:
if __name__ == "__main__":
    main()