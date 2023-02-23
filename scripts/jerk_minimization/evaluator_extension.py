import numpy as np

# Avoidanace Failure Evaluator:
def avoidance_evaluation(self, context) -> np.ndarray:
    # Get Input Port Values and Convert to Numpy Array:
    agent_states = np.asarray(self.get_input_port(self.agent_input).Eval(context)[:2])
    adversary_states = np.asarray(self.get_input_port(self.adversary_input).Eval(context)[:2])

    # 2D Euclidean Distance:
    distance = np.linalg.norm((agent_states - adversary_states))
    eval_distance = distance - self._failure_radius

    # Create Data Point Pairs:
    if eval_distance <= 0:
        x = distance
        y = 1.0
        self._failure_flag = True
    else:
        x = distance
        y = 0.0
        self._failure_flag = False

    return np.array([[x], [y]], dtype=float)

# Tracking Failure Evaluator:
def tracking_evaluation(self, context) -> np.ndarray:
    # Get Input Port Values and Convert to Numpy Array:
    agent_states = np.asarray(self.get_input_port(self.agent_input).Eval(context)[:2])
    adversary_states = np.asarray(self.get_input_port(self.adversary_input).Eval(context)[:2])

    # 2D Euclidean Distance:
    distance = np.linalg.norm((agent_states - adversary_states))
    eval_distance = distance - self._failure_radius

    # Create Data Point Pairs:
    if eval_distance >= 0:
        x = distance
        y = 1.0
        self._failure_flag = True
    else:
        x = distance
        y = 0.0
        self._failure_flag = False

    return np.array([[x], [y]], dtype=float)