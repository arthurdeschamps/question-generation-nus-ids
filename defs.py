import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS_DIR = f"{ROOT_DIR}/models"
NQG_MODEL_DIR = f"{MODELS_DIR}/nqg"
REPEAT_Q_MODEL_DIR = f"{MODELS_DIR}/repeat_q"

DATA_DIR = f"{ROOT_DIR}/data"
PROCESSED_DATA_DIR = f"{DATA_DIR}/processed"
NQG_DATA_HOME = f"{PROCESSED_DATA_DIR}/nqg"

SQUAD_DIR = f"{DATA_DIR}/squad_dataset"
SQUAD_TRAIN = f"{SQUAD_DIR}/train-v1.1.json"
SQUAD_DEV = f"{SQUAD_DIR}/dev-v1.1.json"

SQUAD_REWRITE_MTURK_DIR = f"{SQUAD_DIR}/mturk"
SQUAD_REWRITES_SYNTHETIC_JSON = f"{SQUAD_DIR}/synthetic.train.json"

MEDQUAD_DIR = f"{DATA_DIR}/medquad_dataset"
MEDQUAD_RAW_DIR = f"{MEDQUAD_DIR}/MedQuAD_raw"
MEDQUAD_TRAIN = f"{MEDQUAD_DIR}/train.csv"
MEDQUAD_DEV = f"{MEDQUAD_DIR}/dev.csv"

MEDQA_HANDMADE_DIR = f"{DATA_DIR}/medqa_handmade_dataset"
MEDQA_HANDMADE_RAW_DATASET_FILEPATH = f"{MEDQA_HANDMADE_DIR}/ds_raw.csv"
MEDQA_HANDMADE_FILEPATH = f"{MEDQA_HANDMADE_DIR}/test.csv"
MEDQA_HANDMADE_FOR_NQG_DATASET = f"{NQG_DATA_HOME}/medqa_handmade"

HOTPOT_QA_DATA_DIR = f"{DATA_DIR}/hotpotqa"
HOTPOT_QA_DEV_JSON = f"{HOTPOT_QA_DATA_DIR}/hotpot_dev.json"
HOTPOT_QA_DEV_TARGETS_PATH = f"{HOTPOT_QA_DATA_DIR}/hotpot_dev_targets.txt"

RESULTS_DIR = f"{DATA_DIR}/results"
NQG_PRED_DIR = f"{RESULTS_DIR}/nqg"
NQG_SQUAD_GA_PREDS_OUTPUT_PATH = f"{NQG_PRED_DIR}/squad_GA_preds.txt"
NQG_SQUAD_NA_PREDS_OUTPUT_PATH = f"{NQG_PRED_DIR}/squad_NA_preds.txt"
NQG_SQUAD_NER_PREDS_OUTPUT_PATH = f"{NQG_PRED_DIR}/squad_+NER_preds.txt"
NQG_SQUAD_PREDS_OUTPUT_PATH = f"{NQG_PRED_DIR}/squad_preds.txt"
NQG_MEDQUAD_PREDS_OUTPUT_PATH = f"{NQG_PRED_DIR}/medquad_preds.txt"
NQG_MEDQA_HANDMADE_PREDS_OUTPUT_PATH = f"{NQG_PRED_DIR}/medqa_handmade_preds.txt"
NQG_SQUAD_TESTGA_PREDS_OUTPUT_PATH = f"{NQG_PRED_DIR}/squad_test_GA_preds.txt"
ASS2S_PRED_DIR = f"{RESULTS_DIR}/ass2s"

NQG_MEDQUAD_DATASET = f"{NQG_DATA_HOME}/medquad"
NQG_SQUAD_DATASET = f"{NQG_DATA_HOME}/squad"
NQG_SQUAD_NER_DATASET = f"{NQG_DATA_HOME}/squad_+NER"

TRAINED_MODELS_DIR = f"{MODELS_DIR}/trained"
PRETRAINED_MODELS_DIR = f"{MODELS_DIR}/pre_trained"

LOGS_DIR = f"{TRAINED_MODELS_DIR}/logs"
GRADIENT_DIR = f"{LOGS_DIR}/gradient_tape"

GKG_API_KEY_FILEPATH = f"{ROOT_DIR}/.gkg_api_key"
GKG_SERVICE_URL = "https://kgsearch.googleapis.com/v1/entities:search"

SG_DQG_DIR = f"{MODELS_DIR}/SG-Deep-Question-Generation"
SG_DQG_DATA = f"{PROCESSED_DATA_DIR}/sg_dqg"
SG_DQG_SQUAD_DATA = f"{SG_DQG_DATA}/squad"
SG_DQG_SQUAD_DEBUG_DATA = f"{SG_DQG_DATA}/squad_debug"

SG_DQG_HOTPOT_PREDS_PATH = f"{RESULTS_DIR}/sg_dqg/hotpot_preds.txt"
SG_DQG_SQUAD_PREDS_PATH = f"{RESULTS_DIR}/sg_dqg/sg_dqg_squad_preds.txt"

GLOVE_PATH = f"{PRETRAINED_MODELS_DIR}/glove.840B.300d.txt"

ASS2S_DIR = f"{MODELS_DIR}/NQG_ASs2s"
ASS2S_PROVIDED_PROCESSED_DATA_DIR = f"{ASS2S_DIR}/data/processed/mpqg_substitute_a_vocab_include_a"
ASS2S_PROCESSED_DIR = f"{PROCESSED_DATA_DIR}/ass2s"
ASS2S_PROCESSED_SQUAD_DIR = f"{ASS2S_PROCESSED_DIR}/squad/mpqg_substitute_a_vocab_include_a"
ASS2S_PROCESSED_SQUAD_MPQG_DATA = f"{ASS2S_PROCESSED_DIR}/squad/mpqg_data"

# RepeatQ related definitions
REPEAT_Q_DATA_DIR = f"{PROCESSED_DATA_DIR}/repeat_q"
REPEAT_Q_RAW_DATASETS = f"{REPEAT_Q_DATA_DIR}/raw_datasets"
PAD_TOKEN = "<blank>"
UNKNOWN_TOKEN = "<unk>"
EOS_TOKEN = "<eos>"

REPEAT_Q_PREDS_OUTPUT_DIR = f"{RESULTS_DIR}/repeat_q"
REPEAT_Q_SQUAD_OUTPUT_FILEPATH = f"{REPEAT_Q_PREDS_OUTPUT_DIR}/prediction.txt"
REPEAT_Q_SQUAD_DATA_DIR = f"{REPEAT_Q_DATA_DIR}/squad"
REPEAT_Q_EMBEDDINGS_FILENAME = "embeddings.npy"
REPEAT_Q_VOCABULARY_FILENAME = "vocabulary.txt"
REPEAT_Q_FEATURE_VOCABULARY_FILENAME = "feature_vocabulary.txt"
REPEAT_Q_TRAIN_CHECKPOINTS_DIR = f"{TRAINED_MODELS_DIR}/repeat_q"
