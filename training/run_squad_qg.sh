#!/bin/bash

set -x

DATAHOME=${@:(-4):1}
EXEHOME=${@:(-3):1}

SAVEPATH=${@:(-2):1}
ROOT_DIR=${@:(-1):1}

mkdir -p ${SAVEPATH}

cd ${EXEHOME}

python3 train.py \
       -save_path ${SAVEPATH} -log_home ${SAVEPATH} \
       -online_process_data \
       -train_src ${DATAHOME}/train/data.txt.source.txt -src_vocab ${DATAHOME}/train/vocab.txt.pruned \
       -train_bio ${DATAHOME}/train/data.txt.bio -bio_vocab ${DATAHOME}/train/bio.vocab.txt \
       -train_feats ${DATAHOME}/train/data.txt.pos ${DATAHOME}/train/data.txt.ner ${DATAHOME}/train/data.txt.case \
       -feat_vocab ${DATAHOME}/train/feat.vocab.txt \
       -train_tgt ${DATAHOME}/train/data.txt.target.txt -tgt_vocab ${DATAHOME}/train/vocab.txt.pruned \
       -layers 1 \
       -enc_rnn_size 512 -brnn \
       -word_vec_size 300 \
       -dropout 0.5 \
       -batch_size 64 \
       -beam_size 5 \
       -epochs 20 -optim adam -learning_rate 0.001 \
       -curriculum 0 -extra_shuffle \
       -start_eval_batch 1000 -eval_per_batch 500 -halve_lr_bad_count 3 \
       -seed 12345 -cuda_seed 12345 \
       -log_interval 100 \
       -dev_input_src ${DATAHOME}/dev/data.txt.source.txt \
       -dev_bio ${DATAHOME}/dev/data.txt.bio \
       -dev_feats ${DATAHOME}/dev/data.txt.pos ${DATAHOME}/dev/data.txt.ner ${DATAHOME}/dev/data.txt.case \
       -dev_ref ${DATAHOME}/dev/data.txt.target.txt \
       -max_sent_length 100 \
       -gpus 0

