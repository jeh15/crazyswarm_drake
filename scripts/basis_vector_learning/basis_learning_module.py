from functools import partial

import numpy as np
import ml_collections

from pydrake.common.value import Value
from pydrake.systems.framework import (
    LeafSystem,
    PublishEvent,
    TriggerType,
    BasicVector_,
)

import pdb


class BasisLearningNamespace():
    def __init__(self):
        return NotImplementedError()

    def get_basis_vector(self, data: np.ndarray) -> np.ndarray:
        # Create mask for labeled data:
        success_mask = data[:, -1] == 0
        data_success = data[success_mask, :-1]
        data_failure = data[~success_mask, :-1]
        # Find centroids:
        centroid_success = np.mean(data_success, axis=0)
        centroid_failure = np.mean(data_failure, axis=0)
        # Find basis vector from centroids:
        basis_vector = (centroid_failure - centroid_success) / np.linalg.norm(centroid_failure - centroid_success)
        # Project data onto basis vector:
        # data_proj = element_wise_dot(data, basis_vector) / dot(basis_vector, basis_vector)
        # Tianze's note: this is just a rotation problem. Rotate such that x tells the projection and y tells the distance
        data_projected = np.einsum('ij,j->i', data[:, :-1], basis_vector)
        return data_projected

    # Override evaulation function
    def evaluate(self):
