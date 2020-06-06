import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

NQG_MODEL_DIR = f"{ROOT_DIR}/models/nqg"

DATA_DIR = f"{ROOT_DIR}/data"
PROCESSED_DATA_DIR = f"{DATA_DIR}/processed"
NQG_DATA_HOME = f"{PROCESSED_DATA_DIR}/nqg"

SQUAD_DIR = f"{DATA_DIR}/squad_dataset"
SQUAD_TRAIN = f"{SQUAD_DIR}/train-v1.1.json"
SQUAD_DEV = f"{SQUAD_DIR}/dev-v1.1.json"

MEDQUAD_DIR = f"{DATA_DIR}/medquad_dataset"
MEDQUAD_RAW_DIR = f"{MEDQUAD_DIR}/MedQuAD_raw"
MEDQUAD_TRAIN = f"{MEDQUAD_DIR}/train.csv"
MEDQUAD_DEV = f"{MEDQUAD_DIR}/dev.csv"

MEDQA_HANDMADE_DIR = f"{DATA_DIR}/medqa_handmade_dataset"
MEDQA_HANDMADE_RAW_DATASET_FILEPATH = f"{MEDQA_HANDMADE_DIR}/ds_raw.csv"
MEDQA_HANDMADE_FILEPATH = f"{MEDQA_HANDMADE_DIR}/test.csv"
MEDQA_HANDMADE_FOR_NQG_DATASET = f"{NQG_DATA_HOME}/medqa_handmade"

RESULTS_DIR = f"{DATA_DIR}/results"
NQG_SQUAD_GA_PREDS_OUTPUT_PATH = f"{RESULTS_DIR}/nqg/squad_GA_preds.txt"
NQG_SQUAD_NA_PREDS_OUTPUT_PATH = f"{RESULTS_DIR}/nqg/squad_NA_preds.txt"
NQG_SQUAD_NER_PREDS_OUTPUT_PATH = f"{RESULTS_DIR}/nqg/squad_+NER_preds.txt"
NQG_SQUAD_PREDS_OUTPUT_PATH = f"{RESULTS_DIR}/nqg/squad_preds.txt"
NQG_MEDQUAD_PREDS_OUTPUT_PATH = f"{RESULTS_DIR}/nqg/medquad_preds.txt"
NQG_MEDQA_HANDMADE_PREDS_OUTPUT_PATH = f"{RESULTS_DIR}/nqg/medqa_handmade_preds.txt"
NQG_SQUAD_TESTGA_PREDS_OUTPUT_PATH = f"{RESULTS_DIR}/nqg/squad_test_GA_preds.txt"

NQG_MEDQUAD_DATASET = f"{NQG_DATA_HOME}/medquad"
NQG_SQUAD_DATASET = f"{NQG_DATA_HOME}/squad"
NQG_SQUAD_NER_DATASET = f"{NQG_DATA_HOME}/squad_+NER"

TRAINED_MODELS_DIR = f"{ROOT_DIR}/models/trained"
PRETRAINED_MODELS_DIR = f"{ROOT_DIR}/models/pre_trained"

LOGS_DIR = f"{TRAINED_MODELS_DIR}/logs"
GRADIENT_DIR = f"{LOGS_DIR}/gradient_tape"

GKG_API_KEY_FILEPATH=f"{ROOT_DIR}/.gkg_api_key"
GKG_SERVICE_URL="https://kgsearch.googleapis.com/v1/entities:search"
