from defs import REPEAT_Q_SQUAD_DATA_DIR, REPEAT_Q_VOCABULARY_FILENAME, REPEAT_Q_FEATURE_VOCABULARY_FILENAME


class ModelConfiguration:

    def __init__(self,
                 nb_epochs=20,
                 dropout_rate=0.3,
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
                 data_dir=REPEAT_Q_SQUAD_DATA_DIR,
                 restore_supervised_checkpoint=False,
                 supervised_model_checkpoint_path=None,
                 supervised_epochs=6,
                 dev_step_size=100,
                 learning_rate=None,
                 saving_model=False,
                 training_beam_search_size=5,
                 nb_episodes=32,
                 use_ner_features=True,
                 use_pos_features=True):
        super(ModelConfiguration, self).__init__()
        self.recurrent_dropout = recurrent_dropout
        self.dropout_rate = dropout_rate
        self.fact_encoder_hidden_size = fact_encoder_hidden_size
        self.base_question_encoder_hidden_size = base_question_encoder_hidden_size
        self.embeddings_pretrained = embeddings_pretrained
        self.embedding_size = embedding_size
        self.data_dir = data_dir
        self.vocabulary_path = f"{data_dir}/{REPEAT_Q_VOCABULARY_FILENAME}"
        self.feature_vocabulary_path = f"{data_dir}/{REPEAT_Q_FEATURE_VOCABULARY_FILENAME}"
        self.question_attention = question_attention_function
        self.facts_attention = facts_attention_function
        self.attention_depth = attention_depth
        self.decoder_hidden_size = decoder_hidden_size
        self.max_generated_question_length = max_generated_question_length
        self.decoder_readout_size = decoder_readout_size
        self.batch_size = batch_size
        self.restore_supervised_checkpoint = restore_supervised_checkpoint
        self.supervised_model_checkpoint_path = supervised_model_checkpoint_path
        self.supervised_epochs = supervised_epochs
        self.epochs = nb_epochs
        self.dev_step_size = dev_step_size
        self.learning_rate = learning_rate
        self.saving_model = saving_model
        self.training_beam_search_size = training_beam_search_size
        self.nb_episodes = nb_episodes
        self.use_ner_features = use_ner_features
        self.use_pos_features = use_pos_features

    @staticmethod
    def new() -> 'ModelConfiguration':
        config = ModelConfiguration()
        return config

    def with_ner_features(self, use_ner_features):
        self.use_ner_features = use_ner_features
        return self

    def with_pos_features(self, use_pos_features):
        self.use_pos_features = use_pos_features
        return self

    def with_episodes(self, nb_episodes):
        self.nb_episodes = nb_episodes
        return self

    def with_dropout_rate(self, dropout_rate):
        self.dropout_rate = dropout_rate
        return self

    def with_recurrent_dropout(self, recurrent_dropout):
        self.recurrent_dropout = recurrent_dropout
        return self

    def with_batch_size(self, batch_size):
        self.batch_size = batch_size
        return self

    def with_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate
        return self

    def with_restore_supervised_checkpoint(self):
        self.restore_supervised_checkpoint = True
        return self

    def with_supervised_model_checkpoint_path(self, ckpt_path):
        self.supervised_model_checkpoint_path = ckpt_path
        return self

    def with_saving_model(self, save_model: bool):
        self.saving_model = save_model
        return self

    def with_supervised_epochs(self, nb_epochs):
        self.supervised_epochs = nb_epochs
        return self

    def with_epochs(self, nb_epochs):
        self.epochs = nb_epochs
        return self

    def with_dev_step_size(self, dev_step_size):
        self.dev_step_size = dev_step_size
        return self

    def __str__(self):
        str_builder = ""
        for param_name, default_value in self.__dict__.items():
            str_builder += f"{param_name}: {default_value}\n"
        return str_builder
