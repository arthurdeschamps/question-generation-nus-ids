import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

BERT_DIR = f"{ROOT_DIR}/bert"
BERT_BASE_UNCASED = f"{BERT_DIR}/models/uncased_L-12_H-768_A-12"

NQG_DIR = f"{ROOT_DIR}/nqg"

SQUAD_DIR = f"{ROOT_DIR}/squad_2_dataset"
SQUAD_TRAIN = f"{SQUAD_DIR}/train-v2.0.json"
SQUAD_DEV = f"{SQUAD_DIR}/dev-v2.0.json"
