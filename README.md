# question-generation-nus-ids

## Using the NQG model

To generate questions for SQuAD dataset using the NQG model, run the following command:

`python models/seq2seq.py --model_path
/models/pre_trained/nqg/data/redistribute/QG/models/NQG_plus/your_model.pt`

## Evaluate your predictions
First off, make sure your prediction file contains one generated question per line.
Then, to compute the f1 score and the number of perfect predictions, run:

``