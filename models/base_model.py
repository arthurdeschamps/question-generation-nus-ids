from abc import ABC
from tensorflow.keras import Model


class BaseModel(Model, ABC):
    """
    Model to inherit from to create a model within this project.
    """

    def __init__(self, *args, **kwargs):
        super(BaseModel, self).__init__(*args, **kwargs)
        self.embedder = None
        self.model = None
        self.max_sequence_length = None
