from abc import ABC
from tensorflow.keras import Model

from data_utils.embeddings import Embedder


class BaseModel(Model, ABC):
    """
    Model to inherit from to create a model within this project.
    """

    def __init__(self, embedder: Embedder, model, max_sequence_length, *args, **kwargs):
        super(BaseModel, self).__init__(*args, **kwargs)
        self.embedder = embedder
        self.model = model
        self.max_sequence_length = max_sequence_length
