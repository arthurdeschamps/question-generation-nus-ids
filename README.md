# TODOs
- Integrate: SG-DQG, ASs2s and CGC-QG 
- Update this readme once baseline models are setup

# Question Generation Integrating Knowledge Basis
## General Instructions
Please run the following python code:

```[python]
import stanza
stanza.download('en') 
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
```
## How to run RepeatQ
### What you will need to provide
You will need to provide a JSON file containing your whole dataset (train+dev+test) with the following schema:
```javascript
{
    "facts": List[String],
    "base_question": String,
    "target_question": String
}
```
Every string should be tokenized and lower cased. An example showing how to go about 
creating this file can be found in `data_processing.data_generator.generate_repeat_q_squad_raw`.
### Data Processing Step
Next, you will need to run `models.repeat_q` in `preprocessing` mode, passing in argument
the path to the JSON file mentioned above. This will create a vocabulary file and optionally
an embedding matrix file for you.
### Training
You can now train the model using `models.repeat_q` in `training` mode, passing as argument
the folder containing the data files created during the previous step.
## Knowledge Graph API
### Google Knowledge Graph
Please store you api key in a file ".gkg_api_key" located at the root directory 
## Instructions for the NQG model (Seq2Seq)
To pre-process SG DQG data, you'll need to run `python -m spacy download en_core_web_sm` prior to doing anything.

To run anything related to the NQG model, you'll want to use the script models/seq2seq.py.

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

## Instructions for SG-DQG
- Follow instructions 1 from: https://github.com/yanghoonkim/NQG_ASs2s
- Have the GloVe embedding .txt file in /models/pretrained/ and run `models/NQG_ASs2s/data/process_embedding.py`
