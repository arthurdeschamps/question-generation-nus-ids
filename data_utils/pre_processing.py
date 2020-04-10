from typing import List
import numpy as np
import tensorflow as tf


def pad_data(data: List[np.ndarray], padding_value) -> List[tf.Tensor]:
    paddings = np.array([0, np.max(list(datapoint.shape[0] for datapoint in data))]).reshape((1, -1))
    return list(
        tf.pad(datapoint, paddings=paddings - np.array((0, len(datapoint))), mode='CONSTANT',
               constant_values=padding_value) for datapoint in data
    )
