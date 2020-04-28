# NQG_Interrogative_Phrases

This repository is the implementation of the paper of Neural Question Generation using interrogative phrases at INLG 2019.
This code is based on [the OpenNMT project](https://github.com/OpenNMT/OpenNMT) and the [Pytorch-BERT](https://github.com/huggingface/pytorch-transformers)

Notice: We use torch==1.1.0 and cuda==9.1.85 . If your cuda (or cudnn) version is different from it, this code may not work due to the compatibility with pytorch.

## Requirement

    pip install -r requirements.txt

## How to use

### Generate questions

0.Corenlp

This codes use [Standford CoreNLP](https://stanfordnlp.github.io/CoreNLP/) for preprocess. Please download the link and activate it by the below command

    java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 30000

1.download the data

    wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json -O data/squad-train-v1.1.json
    wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -O data/squad-dev-v1.1.json
    wget http://nlp.stanford.edu/data/glove.840B.300d.zip -P data/
    unzip data/glove.840B.300d.zip -d data/

2.preprocess data

    python qg_prepro_corenlp.py

    python qg_process.py

    python preprocess.py \
    -train_src data/squad-src-train-interro-repanswer.txt \
    -train_tgt data/squad-tgt-train-interro-repanswer.txt \
    -valid_src data/squad-src-val-interro-repanswer.txt \
    -valid_tgt data/squad-tgt-val-interro-repanswer.txt \
    -save_data data/demo \
    -lower -dynamic_dict

    python embeddings_to_torch.py \
    -emb_file_both "data/glove.840B.300d.txt" \
    -dict_file "data/demo.vocab.pt" \
    -output_file "data/embeddings"

3.Train

    python3 train.py \
    -data data/demo \
    -save_model model_data/demo-model \
    -pre_word_vecs_enc "data/embeddings.enc.pt" -pre_word_vecs_dec "data/embeddings.dec.pt" \
    -fix_word_vecs_enc -fix_word_vecs_dec \
    -copy_attn -copy_attn_type general -coverage_attn -lambda_coverage 1 \
    -reuse_copy_attn -copy_loss_by_seqlength \
    -gpu_ranks 3 -world_size 1 \
    -optim sgd -learning_rate 1 -learning_rate_decay 0.5 -start_decay_steps 20000 -decay_steps 2500 -train_steps 50000

4.Generate

    python translate.py \
    -src data/squad-src-test-interro-repanswer.txt \
    -output data/squad-pred-test-interro-repanswer.txt \
    -replace_unk -dynamic_dict \
    -length_penalty avg \
    -model model_data/<model_name>

5.Evaluate

    python2 qgevalcap/eval.py \
    -src data/squad-src-test-interro-repanswer.txt \
    -tgt data/squad-tgt-test-interro-repanswer.txt \
    -out data/squad-pred-test-interro-repanswer.txt

If you want the result without interrogative answer,

    python bleu_noninterro.py \
    --src data/squad-src-test-interro-repanswer.txt \
    --tgt data/squad-tgt-test-interro-repanswer.txt \
    --pred data/squad-pred-test-interro-repanswer.txt \
    --interro data/squad-interro-test-interro-repanswer.txt \
    --noninterro data/squad-noninterro-test-interro-repanswer.txt \
    --p_noninterro data/squad-p_noninterro-test-interro-repanswer.txt \
    --each_interro

    python2 qgevalcap/eval.py \
    -src data/squad-src-test-interro-repanswer.txt \
    -tgt data/squad-noninterro-test-interro-repanswer.txt \
    -out data/squad-p_noninterro-test-interro-repanswer.txt

### Question Answering test for BERT model

1.Generate questions

    python translate.py \
    -src data/squad-src-train-interro-repanswer.txt \
    -output data/squad-pred-train-interro-repanswer.txt \
    -replace_unk -dynamic_dict \
    -stepwise_penalty \
    -length_penalty avg \
    -coverage_penalty summary -beta -1 \
    -model model_data/<model_name>

    python translate.py \
    -src data/squad-src-test-interro-repanswer.txt \
    -output data/squad-pred-test-interro-repanswer.txt \
    -replace_unk -dynamic_dict \
    -stepwise_penalty \
    -length_penalty avg \
    -coverage_penalty summary -beta -1 \
    -model model_data/<model_name>

2.preprocess

    python bert_prepro_modify.py \
    --modify_path_train data/squad-pred-train-interro-repanswer.txt \
    --modify_path_dev data/squad-pred-test-interro-repanswer.txt \
    --output_name interro-repanswer \
    --modify

3.train the model and predict the answer

    mkdir data/interro-repanswer-modify/

    python ./run_squad.py \
      --bert_model bert-base-uncased \
      --do_train \
      --do_predict \
      --do_lower_case \
      --output_dir data/interro-repanswer-modify/ \
      --train_file data/squad-train-interro-repanswer-modify.json \
      --predict_file data/squad-dev-interro-repanswer-modify.json \
      --learning_rate 3e-5 \
      --num_train_epochs 2 \
      --max_seq_length 384 \
      --doc_stride 128 \
      --train_batch_size 12 \
      --gradient_accumulation_steps 2 \
      --gpu_id 2 --log_file data/log_file.txt

4.evaluate generated questions

    python evaluate.py \
    --dataset_file data/squad-dev-interro-repanswer-modify.json \
    --prediction_file data/interro-repanswer-modify/predictions.json
