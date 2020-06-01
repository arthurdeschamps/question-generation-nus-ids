# TODOs
- Integrate: SG-DQG, ASs2s and CGC-QG 
- Update this readme once baseline models are setup

# Question Generation Integrating Knowledge Basis
## Instructions for the NQG model (Seq2Seq)
To run anything related to the NQG model, you'll want to use the script models/seq2seq.py.

To pre-process SG DQG data, you'll need to run `python -m spacy download en_core_web_sm` prior to doing anything.
### Train
**Command**: `seq2seq.py train`

**Description**: 
Trains the model using data located at /data/processed/nqg. This directory shall contain
two subdirectories "dev" and "train". The content of these directories shall follow
the format used by the original NQG team: https://res.qyzhou.me/redistribute.zip, even though
the original dataset can be any of your liking and the NER and POS
features can be modified as well to use any convention/tool.

**Options**:

`--vocab_size num` : prune the vocabulary of the dataset to the required number of words "num". Optional; Default value: 
20000

### Generating data
**Command**: `seq2seq.py generate_data`

**Description**: Generates the necessary data from the raw SQuAD dataset to train the NQG model on. The SQuAD 1.1 data
files shall be stored at /data/squad_dataset.

### Make predictions
**Command**: Use `translate.py` from the [NQG repository](https://github.com/magic282/NQG) with any .pt file storing
your trained model.

### Make predictions (NQG+ SQuAD dev set)
**Command**: `seq2seq.py beam_search`

**Description**: Makes predictions for the SQuAD dev set (see data format from section **train**).

**Options**:

`--model_path path` : path to a .pt trained model file.

