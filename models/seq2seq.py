from typing import List

from tensorflow.keras import Model


class NQG(Model):
    def generate_questions(self, tokens: List[int]):
        pass

    def call(self, inputs, training=None, mask=None):
        raise NotImplementedError()
