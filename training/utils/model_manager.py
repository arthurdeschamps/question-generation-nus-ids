import time
import tensorflow as tf
from defs import TRAINED_MODELS_DIR, PRETRAINED_MODELS_DIR


class ModelManager:
    """
    Use this class to save and load keras models.
    """

    @staticmethod
    def save_model(model: tf.keras.Model, model_name=None):
        if model_name is None:
            model_name = time.strftime("%Y%m%d-%H%M%S")
        model.save_weights(ModelManager._model_filepath(model_name))

    @staticmethod
    def load_model(model: tf.keras.Model, model_name):
        model.load_weights(ModelManager._model_filepath(model_name))
        return model

    @staticmethod
    def load_pretrained_model(model, model_name):
        model.load_weights(f"{PRETRAINED_MODELS_DIR}/{model_name}")
        return model

    @staticmethod
    def _model_filepath(model_name):
        return f"{TRAINED_MODELS_DIR}/{model_name}/{model_name}"
