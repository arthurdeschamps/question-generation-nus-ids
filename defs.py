import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

NQG_DIR = f"{ROOT_DIR}/models/nqg"

DATA_DIR = f"{ROOT_DIR}/data"
PROCESSED_DATA_DIR = f"{DATA_DIR}/processed"
SQUAD_DIR = f"{DATA_DIR}/squad_2_dataset"
SQUAD_TRAIN = f"{SQUAD_DIR}/train-v2.0.json"
SQUAD_DEV = f"{SQUAD_DIR}/dev-v2.0.json"

RESULTS_DIR = f"{DATA_DIR}/results"
NQG_PREDS_OUTPUT_PATH = f"{RESULTS_DIR}/nqg/preds.txt"

TRAINED_MODELS_DIR = f"{ROOT_DIR}/models/trained"
PRETRAINED_MODELS_DIR = f"{ROOT_DIR}/models/pre_trained"

LOGS_DIR = f"{TRAINED_MODELS_DIR}/logs"
GRADIENT_DIR = f"{LOGS_DIR}/gradient_tape"
