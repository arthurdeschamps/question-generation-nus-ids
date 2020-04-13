from abc import ABC
from tensorflow.keras import Model


class BaseModel(Model, ABC):

    def __init__(self, *args, **kwargs):
        super(BaseModel, self).__init__(*args, **kwargs)
        self.embedder = None
        self.model = None
        self.max_sequence_length = None
