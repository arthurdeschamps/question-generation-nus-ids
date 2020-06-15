from logging import warning

import numpy as np
import pickle as pkl
import os

import psutil
from tqdm import tqdm

from defs import PRETRAINED_MODELS_DIR, GLOVE_PATH


def run(data_dir):
    print('########## Embedding processing start ##########\n')

    # SETTINGS >>>

    dic_name = 'vocab.dic'
    embedding_name = 'glove840b_vocab300.npy'

    glove = 'glove.840B.300d'

    # SETTINGS <<<


    # LOAD & PROCESS GloVe >>>
    if not os.path.exists(data_dir + '/' + glove + '.dic.npy'):
        # Load GloVe
        print('>>> Reading GloVe file...', end=' ')
        f = open(GLOVE_PATH, encoding='utf-8')
        lines = f.readlines()
        f.close()
        print('Complete\n')

        # Process GloVe
        print('>>> Processing Glove...')
        embedding = dict()
        for i, line in tqdm(enumerate(lines)):
            if i % 50000 == 0 and psutil.virtual_memory().percent > 70:
                warning("Couldn't load full GloVe embedding file due to memory overload.")
                break
            splited = line.split()
            try:
                embedding[splited[0]] = list(map(float, splited[1:]))
            except ValueError:
                print(line)
                pass

        # Save processed GloVe as dic file
        print('>>> Saving processed GloVe...')
        np.save(data_dir + '/' + glove + '.dic', embedding, allow_pickle=True)
        print('Complete\n')
    else:
        print('Processed GloVe exists!')
        print('>>> Loading processed GloVe file...')
        embedding = np.load(data_dir + '/' + glove + '.dic.npy', allow_pickle=True).item()
        print('Complete\n')

    # LOAD & PROCESS GloVe <<<


    # PRODUCE PRE-TRAINED EMBEDDING >>>


    # Load vocabulary
    print('>>> Loading vocabulary...', end=' ')
    with open(os.path.join(data_dir + "/mpqg_substitute_a_vocab_include_a", dic_name), mode='rb') as f:
        vocab = pkl.load(f)
    print('Complete\n')

    # Initialize random embedding and extract pre-trained embedding
    print('>>> Producing pre-trained embedding...', end=' ')
    embedding_vocab = np.random.ranf((len(vocab), 300)) - np.random.ranf((len(vocab), 300))

    embedding_vocab[0] = 0.0  # vocab['<PAD>'] = 0
    embedding_vocab[1] = embedding['<s>']  # vocab['<GO>'] = 1
    embedding_vocab[2] = embedding['EOS']  # vocab['<EOS>'] = 2
    embedding_vocab[3] = embedding['UNKNOWN']  # vocab['<UNK>'] = 3

    unk_num = 0
    for word, idx in list(vocab.items()):
        if word in embedding:
            embedding_vocab[idx] = embedding[word]
        else:
            unk_num += 1
    print('Complete\n')

    # Save embedding
    print('>>> Saving pre-trained embedding', end=' ')
    np.save(os.path.join(data_dir + "/mpqg_substitute_a_vocab_include_a", embedding_name), embedding_vocab)
    print('Complete\n')

    print('---------- Statistics ----------')
    print('Vocabulary size: %d' % len(embedding_vocab))
    print('Unknown words: %d' % unk_num)

    # PRODUCE PRE-TRAINED EMBEDDING <<<

