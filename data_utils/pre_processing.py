from typing import List
import numpy as np
import tensorflow as tf

from data_utils.class_defs import SquadExample, Paragraph


def pad_data(data: List[np.ndarray], padding_value) -> List[tf.Tensor]:
    """
    Transforms a variable sized list of arrays to a rectangular array by padding the arrays accordingly with the
    given padding value.
    """
    paddings = np.array([0, np.max(list(datapoint.shape[0] for datapoint in data))]).reshape((1, -1))
    return list(
        tf.pad(datapoint, paddings=paddings - np.array((0, len(datapoint))), mode='CONSTANT',
               constant_values=padding_value) for datapoint in data
    )


def remove_question_mark(text: str) -> str:
    return text.replace('?', '')
