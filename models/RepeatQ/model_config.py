from logging import info

from defs import NQG_SQUAD_DATASET, REPEAT_Q_SQUAD_DATA_DIR, REPEAT_Q_VOCABULARY_FILENAME


class ModelConfiguration:

    def __init__(self,
                 encoder_recurrent_dropout=0.1,
                 fact_encoder_hidden_size=512,
                 base_question_encoder_hidden_size=512,
                 embeddings_pretrained=True,
                 embedding_size=None,
                 data_dir=REPEAT_Q_SQUAD_DATA_DIR):
        super(ModelConfiguration, self).__init__()
        self.encoder_recurrent_dropout = encoder_recurrent_dropout
        self.fact_encoder_hidden_size = fact_encoder_hidden_size
        self.base_question_encoder_hidden_size = base_question_encoder_hidden_size
        self.embeddings_pretrained = embeddings_pretrained
        self.embedding_size = embedding_size
        self.data_dir = data_dir
        self.vocabulary_path = f"{data_dir}/{REPEAT_Q_VOCABULARY_FILENAME}"

    @staticmethod
    def build_config(config) -> 'ModelConfiguration':
        config = ModelConfiguration(**config)
        info("\nModel Parameters:\n--------------------------\n" +
             str(config) +
             "--------------------------\n")
        return config

    def __str__(self):
        str_builder = ""
        for param_name, default_value in self.__dict__.items():
            str_builder += f"{param_name}: {default_value}\n"
        return str_builder
