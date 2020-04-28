# Question Generation Integrating Knowledge Basis
## Instructions for the NQG model (Seq2Seq)
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

### Make predictions (SQuAD dev set)
**Command**: `seq2seq.py predict`

**Description**: Makes predictions for the SQuAD dev set (see data format from section **train**).

**Options**:

`--model_path path` : path to a .pt trained model file.

## Literature Review
### BLEU Scores
Here are reported the BLEU-4 scores on the SQuAD dataset for each architecture presented down below.

| Model Number        | BLEU score           | Code Available  |
| :-------------: |:-------------:| :-----:|
| 1 | 15.64 | N |
| 2  | 14.39 (?)  | N |
| 3 | 16.31  | N |
| 4 | 15.29 | N |
| 5 | 16.99 | Y |
| 6 |16.27 | N |
| 7 | 18.53 | N |
| 8 | ? | Y |
| 9 | 13.29 | Y |


### 1. Answer-focused and Position-aware Neural Question Generation
**Link**: https://www.aclweb.org/anthology/D18-1427.pdf

**Goal**: Solving 2 common issues: mismatched question types and unrelated copied words (words taken from the passage that are irrelevant or misleading in answering the question).

**Idea**: A standard LSTM based encoder-decoder network is modified with an additional interrogative word generation probability, used at every step of the generation process. They also try to focus the attention of nearby context words, as opposed to distant words, to avoid copying unrelated words.

**Best model**: The baseline model is an encoder-decoder pointer-based network with attention mechanism, where the encoder is a bi-directional LSTM network and the decoder a unidirectional LSTM. The decoder is augmented with three “probability switches” which determine if the word will be copied, generated or if an interrogative word will be drawn from a distribution. This interrogative word distribution is generated from the hidden state generated right before the start of the answer in the context, the  current hidden state and the current attention-weighted context vector. The position-aware single layer network uses relative distance embeddings (relative to the answer) to compute an attention distribution. Both position-aware and non-position aware context vectors are finally combined to create a final word distribution.

**Scores**: 
Squad: BLEU-1: 43.02, BLEU-2: 28.14, BLEU-3: 20.51, BLEU-4: 15.64
Marco: BLEU-1: 48.24, BLEU-2: 35.95, BLEU-3: 25.79, BLEU-4: 19.45

**Datasets**: SQuAD and MARCO

**Code/Dataset available**: no/yes


### 2. Neural Generation of Diverse Questions using Answer Focus, Contextual and Linguistic Features
**Link**: https://www.aclweb.org/anthology/W18-6536.pdf

**Goal**: Integrating different features instead of only the plain text/passage and attempt to use question-oriented sentence embeddings as extra inputs.

**Idea**: They make use of NER, Case and BIO features and resolve co-references beforehand by integrating them directly into the input text. On the neural architectural side, they use a sentence embedding network which takes as input raw sentences and uses the output sentence encoding as input to another network in combination with the feature-rich, answer-focused embeddings.
To produce question-oriented sentence embeddings, they first train the whole model end-to-end, with a sentence encoder being fed the question directly, and then later on train a brand new sentence encoder to maximize the similarity between its encodings and the question-specific encodings. This network is at this point fed with normal passages, as opposed to questions (which obviously wouldn’t be possible during testing or when used on unlabelled data).
They also make use of a copy mechanism.

**Best model**: Bi-directional LSTM for the sentence embedding generator, LSTM for the encoder and LSTM with attention for the decoder.

**Score**s: BLEU-4: 14.39 (not clear if it is an average over BLEU from 1 to 4 or if only 4-grams were looked at).
These scores are to take with a grain of salt as the methodology is different from the one used for calculating the score above and isn’t extremely clear: METEOR: 22.26 ROUGE: 48.23

**Datasets**: SQuAD

**Code/Dataset available**: no/yes

### 3. Question-type Driven Question Generation
**Link**: https://www.aclweb.org/anthology/D19-1622.pdf

**Goal**: Generate the interrogative word first and feed it to the predictive model to improve the correctness of the questions.

**Idea**: One encoder that produces latent representations of the features and words, one to predict the interrogative word (which uses the former’s output vectors as input) and finally one decoder to produce the question.

**Best model**: The feature-rich encoder is a bi-LSTM, the interrogative word generator a simple LSTM and the decoder a simple LSTM with attention on its inputs. The first word fed to the decoder is the predicted interrogative word. They also use a copy mechanism and feed-forward neural network to produce word distributions. The loss function is a combination of both encoder’s loss (joint learning task).

**Scores**: 
SQuAD -> BLEU-1: 43.11 BLEU-2: 29.13 BLEU-3: 21.39 BLEU-4: 16.31 

MARCO -> BLEU-1: 55.67 BLEU-2: 38.16 BLEU-3: 28.12 BLEU-4: 21.59

**Datasets**: SQuAD (they directly use the data given by the NQG team)  and MARCO

**Code/Dataset available**: no/yes

### 4. Generating Highly Relevant Questions
**Link**: https://www.aclweb.org/anthology/D19-1614.pdf

**Goal**: Improving the relevance and quality or questions.

**Idea**: Improve the standard copy mechanism by potentially not copying a whole word but a morphological transformation of it and use a QA model to predict answers from generated questions and compute the F1 score between the gold answer and the generated one to allow re-ranking of most likely generated questions.

**Best model**: Seq2Seq model with partial copy mechanism and QA reranking. These two last features are original ideas from the paper while the former is a model borrowed from https://www.aclweb.org/anthology/N18-2090.pdf.

**Scores**: BLEU-1: 44.61 BLEU-2: 28.78 BLEU-3: 20.59 BLEU-4: 15.29 METEOR: 20.13

**Datasets**: Squad (version not specified)

**Code/Dataset available**: no/yes

### 5. Let’s Ask Again: Refine Network for Automatic Question Generation
**Link**: https://www.aclweb.org/anthology/D19-1326.pdf

**Goal**: Generating more grammatically correct, coherent and to the point questions.

**Idea**: Taking a rough question generated as a first draft and using a second neural model to refine it. They explore 2 options for the refining network; The first one simply performs a second pass to improve the quality of the question whereas the second one incorporates a reward system which allows to enhance specific aspects of the question (fluency, answerability, etc).

**Best model**: 2 encoders, both bi-LSTM networks. The first one encodes the passage (with GLoVe, char and relative to the answer positional embeddings) and the second the answer. The two outputs are then fusioned through a non-linear layer and produces a hidden representation that will be used by the decoder to generate the first question. The decoder generator is an LSTM with attention mechanism. Finally, the refinement decoder takes as input the hidden representation vectors from the passage decoder and the embeddings of the generated question and uses attention of both of these sequences. These two new vectors as well as the final answer hidden representation are fed through yet another LSTM network. 

**Scores**: 
On SQuAD: BLEU-1: 46.41 BLEU-2: 30.66 BLEU-3: 22.42 BLEU-4: 16.99 ROUGE-L: 45.03 METEOR: 21.10

**Datasets**: SQuAD, HOTPOT-QA, and DROP

**Code/Dataset available**: Yes/yes

### 6. Improving Question Generation With to the Point Context
**Link**: https://www.aclweb.org/anthology/D19-1317.pdf

**Goal**: Overcoming the issue of useful context words located far away from the answer span.

**Idea**: Use a linguistic tool (OpenIE) to extract phrase relations (structured text) and feed it along with the unstructured text to their neural model. They select answer-relevant relations by counting the number of overlapping tokens between the answer and the relation or their confidence levels in case tie-breakers are needed.

**Best model**: The neural model is an encoder-decoder network where a gated attention network and a copy mechanism are placed in the middle. Its inputs are the raw passage as well as the structured relations. They both flow through 2 separate encoders and their final latent representations are combined through the gated attention network. They also use NER, POS and BIO features as inputs to their model. All RNNs are LSTM networks (2 layers, 600 hidden units), but encoders are bi-directional while decoders are uni-directional.

**Scores**: BLEU-1: 45.66, BLEU-2: 30.21, BLEU-3: 21.82 BLEU-4: 16.27 METEOR: 20.36 ROUGE-L: 44.35

**Datasets**: SQuAD 1.1 excluding questions which don’t have overlapping non-stop words with their corresponding answer

**Code/Dataset available**: No/Yes

### 7. Let Me Know What to Ask: Interrogative-Word-Aware Question Generation
**Link**: https://www.aclweb.org/anthology/D19-5822.pdf

**Goal**: See the benefits of generating the interrogative word and question in separate steps, the former with a classifier and the latter with a decoder.

**Idea**: 
First predict the interrogative word, feed it to the seq2seq model and this latter decides to use the word or not as its first predicted token. The answer in the paragraph is also indicated with an [ANS] … [ANS] pair of tokens.

**Best model**: 
The 2 models are the following:
The classifier is based on BERT: it uses the [CLS] token + learnable NER embedding (of the answer) and feeds the concatenated vector to a simple feed-forward network. Classification realm: what, which, where, when, who, why, how, and others
Question generator: sequence-to-sequence neural network that uses a gated self-attention in the encoder and an attention mechanism with maxout pointer in the decoder (https://www.aclweb.org/anthology/D18-1424.pdf)
Encoder: takes passage+answer+predicted interrogative word and produces a latent vector. It’s composed of an RNN, a self-attention network and a fusion gate.
Decoder: RNN with attention-layer + copy mechanism. Also uses a maxout pointer mechanism to avoid repeating the same word in the output that is frequent in the input.
During training, the gold interrogative word is fed to the encoder-decoder model.

**Scores**: 47.69 in BLEU-1, 18.53 in BLEU-4, 22.33 in METEOR, and 46.94 in ROUGE-L.

**Datasets**: SQuAD 1.1

**Code/Dataset available**: no/yes

### 8. Neural Question Generation using Interrogative Phrases
**Link**: https://www.aclweb.org/anthology/W19-8613.pdf

**Goal**: See the benefits of generating a question with a specific interrogative word. For instance, if the answer is “yesterday”, we might want to generate a question that starts with “when”. This, according to the authors, improves the quality of the generated questions and allows to have more control over the generated questions.

**Idea**: They modify the data by replacing the answer within its context with a special token <A> and insert directly to its right: <IP> interrogative word <ANS> (the interrogative word is to be extracted manually from the answer). They use an encoder-decoder model for the generation part.
  
**Best model**: Copy mechanism, coverage mechanism (vector indicating how much of the passage has been “covered” is passed and updated after every generated token; this is to avoid under or over-translation). Neural model in an encoder-decoder with attention.

**Scores**: Achieves 18.6 BLEU (the maximum gram length isn’t specified but I assume it to be 4 as they mentioned they did something similar to other papers) score, 20.9 METEOR and 46.9 ROUGE-L on SQuAD 1.1 limited to interrogative phrased questions only.

**Datasets**: Squad 1.1

**Code/Dataset available**: yes/yes

### 9. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
**Link**: https://arxiv.org/pdf/1810.04805.pdf

**Goal**: Generic language model (can be used for a wide range of NLP tasks as a pre-trained model that’s easily extensible).
Before: feature-based (task specific) and fine-tuning (generic models) but unidirectional which is a problem for certain tasks (QA, sentence-level tasks).

**Idea**:
Very general bidirectional model that improves by a good margin results in most of the common NLP tasks. Requires low resources as it is already pretrained. Is more general than previous architectures (e.g. OpenAI GPT) since contexts from both directions can be used for predictions (important in Q/A for instance).
Techniques

Fine-tuning BERT boils down to plugging in input/output pairs where inputs can be:
- (original phrase, paraphrased)
- (premise, hypothesis)
- (question, passage) <- for us
- (phrase, NONE) for text classification or sequence tagging
output is either the output tokens for token level tasks or the [CLS] token for classification

Fine-tuning with small datasets on the large model (>300mio params) might be unstable

**Best Model**:
- Multi-layer bidirectional transformer (to which a usually small layer is added for task specific fine tuning).
- Allows tokens to be sentences or (question, answer) tokens. Final input embedding is: token embedding + segment embedding (A or B) + position embedding (e.g. 1,2, …, 10)
- Uses WordPiece embeddings.

Trained on 2 unsupervised tasks:
- Masked LM: mask random words from the input (15%) and try to predict them
- Next Sentence Prediction: Two sentences, 50% of the time B is the continuation of sentence A, and rest of the time it is a random sentence. Token C is used at the beginning of the input to specify which task is being solved.

Transformer: 
- Network using self-attention and neither recurrent processes or convolutions
- Encoder-decoder architecture
- Encoder has 6 layers composed each of 2 sub-layers: one multi-head self-attention mechanism and one position-wise fully connected FFN
- Decoder is the same as the encoder but each layer contains a a 3rd sub-layer: a multi-head attention over the output of the encoder
- Attention: mult(func(keys, query), values) where func computes a weight vector.
- Multi-head attention: learned linear projections of keys, queries and values, apply attention to each projection, concatenate and finally projected to the output space
- In Transformer model, queries come from earlier decoder layers, values and keys come from the encoder layers’ outputs- 
- Positional encoding: encode the absolute or relative positions of the tokens (necessary since no other spatial information is available such as in a recurrence network or c conv net). They used cos/sin function directly on the position values.


**Datasets**:
BookCorpus (800M words) + English Wikipedia (2’500M words, but only uses sentences, ignoring lists, tables and headers)
Observations 

**Code/Dataset available**: yes/yes

### 10. Neural Question Generation from Text: A Preliminary Study
**Link**: https://arxiv.org/pdf/1704.01792.pdf
**Goal**:
Neural Question Generation servers 2 main purposes: educational and increasing available data for the reverse task (question answering). Up until this paper, no neural approaches to generating questions from unstructured text data had been explored.

**Idea**:
Modification of a seq-2-seq neural network which includes:
- Answer position indicator: answer span in the text (what chunk of text does the generated question answer)
- Lexical features: POS and NER

**Best Model**:
Input vectors are a concatenation of word embeddings, lexical features embeddings and answer position indicator.
The BiGRU model produces one forward and one backward sequence from the input vector. The output vector is the concatenation of these 2. The answer-positional feature is called BIO (B for beginning, I meaning inside the answer and O not part of the answer). The architecture also includes a copy mechanism to copy over rare words from the source text. A probability p is to copy a word t is computed using the decoder state s_t and context vector c_t.

**Scores**: Outperforms the state-of-the-art rule-based question generating system (1.42 -> 2.18 on the human evaluation, 2 being reasonably well formulated and meaningful question, scale goes from 1 to 3). Also improved BLEU-4 score from 9.31 to 13.29.

**Datasets**: SQuAD 1.1.

**Code/Dataset available**: yes/yes

## Miscellaneous
### Question types / Possible answers
\[F\] marks factoid type of questions.
- Where: 
    - Geographical location (country, city, etc) \[F\]
    - Establishment (bank, post office, town hall) \[F\]
    - Type of environment (warm countries, dark rooms, large stadiums) \[F\]
    - Specific but non-geographical places (bathroom, kitchen, basketball court) \[F\]
    
- When:
  - Datetime (specific date + time, December 25th at midnight) \[F\]
  - Recurring time (every Friday, at night) \[F\]
  - Range/period of time (during the Renaissance, during world war 2) \[F\]
  - Future entailment (when my son will be able to walk, when I’ll have a job)
- Why:
  - Reason (Why are you late? because I got lost / because the car stopped) \[F\]
  - Explanation (why do birds have wings? to be able to fly, why do toddlers cry constantly? because they…) \[F\]
- Who:
  - Specific person (Who did this / Who is he? Arthur; Who was the 3rd president of the United States?) \[F\]
  - Type of person/people (who has big calves? cyclists, who can bear the weight of isolation? mentally tough people) \[F\]
- What:
  - Multi-choice question (what is your favorite color? what evaluation metric should we use?) \[F\]
  - Context specific information retrieval (what kind of food do you prefer? What activity would you like to do today? What kind of person is Elizabeth?) \[F\]
  - Open-ended information retrieval (What can I do for you? What did she tell you?)
- How: 
  - Instructional question (How do you bake cookies? How does one replace an exhaust pipe?)
  - Recollection of past events (How did you get here? How did you manage to get an A+?)
  - Rank order scaling (How are you? How spicy is your food? How old are you? How good are you at Boxing?) \[F\]

