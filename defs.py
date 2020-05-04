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
MEDQUAD_FOR_NQG_DATASET = f"{NQG_DATA_HOME}/medquad"

MEDQA_HANDMADE_DIR = f"{DATA_DIR}/medqa_handmade_dataset"
MEDQA_HANDMADE_FILEPATH = f"{MEDQA_HANDMADE_DIR}/test.csv"
MEDQA_HANDMADE_FOR_NQG_DATASET = f"{NQG_DATA_HOME}/medqa_handmade"

RESULTS_DIR = f"{DATA_DIR}/results"
NQG_PREDS_OUTPUT_PATH = f"{RESULTS_DIR}/nqg/preds.txt"
NQG_SQUAD_PREDS_OUTPUT_PATH = f"{RESULTS_DIR}/nqg/squad_preds.txt"
NQG_MEDQUAD_PREDS_OUTPUT_PATH = f"{RESULTS_DIR}/nqg/medquad_preds.txt"
NQG_MEDQA_HANDMADE_PREDS_OUTPUT_PATH = f"{RESULTS_DIR}/nqg/medqa_handmade_preds.txt"

TRAINED_MODELS_DIR = f"{ROOT_DIR}/models/trained"
PRETRAINED_MODELS_DIR = f"{ROOT_DIR}/models/pre_trained"

LOGS_DIR = f"{TRAINED_MODELS_DIR}/logs"
GRADIENT_DIR = f"{LOGS_DIR}/gradient_tape"
