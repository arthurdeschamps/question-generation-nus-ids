from logging import info

from defs import NQG_SQUAD_DATASET, REPEAT_Q_SQUAD_DATA_DIR, REPEAT_Q_VOCABULARY_FILENAME


class ModelConfiguration:

    def __init__(self,
                 recurrent_dropout=0.1,
                 fact_encoder_hidden_size=512,
                 base_question_encoder_hidden_size=512,
                 max_generated_question_length=30,
                 question_attention_function="additive",
                 facts_attention_function="additive",
                 attention_depth=256,
                 embeddings_pretrained=True,
                 embedding_size=300,
                 decoder_hidden_size=512,
                 decoder_readout_size=256,
                 batch_size=32,
                 data_dir=REPEAT_Q_SQUAD_DATA_DIR):
        super(ModelConfiguration, self).__init__()
        self.recurrent_dropout = recurrent_dropout
        self.fact_encoder_hidden_size = fact_encoder_hidden_size
        self.base_question_encoder_hidden_size = base_question_encoder_hidden_size
        self.embeddings_pretrained = embeddings_pretrained
        self.embedding_size = embedding_size
        self.data_dir = data_dir
        self.vocabulary_path = f"{data_dir}/{REPEAT_Q_VOCABULARY_FILENAME}"
        self.question_attention = question_attention_function
        self.facts_attention = facts_attention_function
        self.attention_depth = attention_depth
        self.decoder_hidden_size = decoder_hidden_size
        self.max_generated_question_length = max_generated_question_length
        self.decoder_readout_size = decoder_readout_size
        self.batch_size = batch_size

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
