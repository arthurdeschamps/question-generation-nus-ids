import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

NQG_DIR = f"{ROOT_DIR}/nqg"

SQUAD_DIR = f"{ROOT_DIR}/squad_2_dataset"
SQUAD_TRAIN = f"{SQUAD_DIR}/train-v2.0.json"
SQUAD_DEV = f"{SQUAD_DIR}/dev-v2.0.json"

TRAINED_MODELS_DIR = f"{ROOT_DIR}/models/trained"
PRETRAINED_MODELS_DIR = f"{ROOT_DIR}/models/pre_trained"
