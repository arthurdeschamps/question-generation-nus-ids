import argparse
import codecs
import json
import os
import sys
from logging import warning, info, debug
from pathlib import Path

import spacy
from allennlp.predictors.predictor import Predictor
import allennlp_models.coref  # Required
import allennlp_models.structured_prediction.models.biaffine_dependency_parser  # Required
import numpy as np
from tqdm import tqdm
from data_processing.sg_dqg_dataset import SGDQGDataset
from data_processing.utils import answer_span
from defs import SG_DQG_DIR, GLOVE_PATH, SG_DQG_SQUAD_DATA, SG_DQG_HOTPOT_PREDS_PATH

sys.path.append(f"{SG_DQG_DIR}/build-semantic-graphs")
sys.path.append(f"{SG_DQG_DIR}/src")
# These imports actually work and are necessary
from translate import main as sg_dqg_translate
from merge import merge
from build_semantic_graph import run as build_semantic_graph
from preprocess import main as create_sg_dqg_dataset
from train import run


def json_load(x):
    return json.load(codecs.open(x, 'r', encoding='utf-8'))


def json_dump(d, p):
    return json.dump(d, codecs.open(p, 'w', 'utf-8'), ensure_ascii=False)


def translate(dataset_name):
    if dataset_name == "hotpot":
        sequence_data_path = f"{SG_DQG_DIR}/datasets/preprocessed-data/preprocessed_sequence_data.pt"
        graph_data_path = f"{SG_DQG_DIR}/datasets/preprocessed-data/preprocessed_graph_data.pt"
        valid_data_path = f"{SG_DQG_DIR}/datasets/Datasets/valid_dataset.pt"
    elif dataset_name == "squad":
        sequence_data_path = f"{SG_DQG_SQUAD_DATA}/preprocessed-data/preprocessed_sequence_data.pt"
        graph_data_path = f"{SG_DQG_SQUAD_DATA}/preprocessed-data/preprocessed_graph_data.pt"
        valid_data_path = f"{SG_DQG_SQUAD_DATA}/Datasets/dev_dataset.pt"
    else:
        raise ValueError(f"Unrecognized dataset name: \"{dataset_name}\"")
    translate_args = argparse.Namespace(
        model=f"{SG_DQG_DIR}/models/hotpotqa/generator_best.chkpt",
        sequence_data=sequence_data_path,
        graph_data=graph_data_path,
        valid_data=valid_data_path,
        output=SG_DQG_HOTPOT_PREDS_PATH,
        beam_size=5,
        batch_size=16,
        gpus=[0],
        cuda=True
    )

    sg_dqg_translate(translate_args)


def dependency_parsing(resolved_corefs, updated_bios):
    predictor = Predictor.from_path(
        "https://storage.googleapis.com/allennlp-public-models/biaffine-dependency-parser-ptb-2020.04.06.tar.gz",
        cuda_device=0
    )
    dps = []
    info("Performing dependency parsing...")
    for paragraph_index, evidences in tqdm(enumerate(resolved_corefs)):
        paragraph_bio = updated_bios[paragraph_index]
        dp = []
        bio_index = 0
        for ev_index, evidence in enumerate(evidences):
            pred = predictor.predict(
                sentence=evidence
            )

            nodes = []
            for i in range(len(pred['words'])):
                try:
                    bio = paragraph_bio[bio_index + i]
                except IndexError:
                    warning("Mismatch between bio data length and paragraph length")
                    bio = 0
                nodes.append({
                    'head': pred['predicted_heads'][i] - 1,
                    'pos': pred['pos'][i],
                    'dep': pred['predicted_dependencies'][i],
                    'word': pred['words'][i],
                    'ans':  bio
                })
            bio_index += len(pred['words'])
            dp.append(nodes)
        dps.append(dp)
    return dps


def coreference_resolution(contexts, answers):
    ans_indicators = []
    coreferences = []
    info("Performing coreference resolution...")
    for i, (cr, answer) in tqdm(enumerate(zip(contexts, answers))):
        words = cr['document']
        substitutions = {}
        start_index, end_index = answer_span(words, answer.split(' '))
        if start_index is None or end_index is None:
            warning(f"Answer not found in text: '{answer}'")
            bio = [0 for _ in range(len(words))]
        else:
            bio = [1 if i in range(start_index, end_index+1) else 0 for i in range(len(words))]
        for span_index, span in enumerate(cr['top_spans']):
            # Pronouns have only one word
            if span[1] == span[0] and cr['predicted_antecedents'][span_index] > -1:
                antecedent_span = cr['top_spans'][cr['predicted_antecedents'][span_index]]
                left = antecedent_span[0]
                right = antecedent_span[1]
                substitutions[span[0]] = {
                    "text": words[left:right + 1],
                    "bio": bio[left:right + 1]
                }
        resolved = []
        ans_indicator = []
        for word_index in range(len(words)):
            if word_index in substitutions:
                resolved.extend(substitutions[word_index]['text'])
                ans_indicator.extend(substitutions[word_index]['bio'])
            else:
                resolved.append(words[word_index])
                ans_indicator.append(bio[word_index])
        ans_indicators.append(ans_indicator)
        coreferences.append(resolved)
    return coreferences, ans_indicators


def merge_dps_and_crefs(data_path, dependencies_path, coreferences_path, result_path):
    data = merge(json_load(data_path), json_load(dependencies_path), json_load(coreferences_path))
    json_dump(data, result_path)


def generate_sg_dqg_datasets(dev_or_test="dev"):
    sg_dqg_preprocessed_dir = f"{SG_DQG_SQUAD_DATA}/preprocessed-data/"
    sg_dqg_datasets_dir = f"{SG_DQG_SQUAD_DATA}/Datasets"
    Path(sg_dqg_preprocessed_dir).mkdir(parents=False, exist_ok=True)
    Path(sg_dqg_datasets_dir).mkdir(parents=False, exist_ok=True)

    preprocess_args = argparse.Namespace(
        # train_src=f"{SG_DQG_DIR}/datasets/text-data/train.src.txt",
        # train_tgt=f"{SG_DQG_DIR}/datasets/text-data/train.tgt.txt",
        # valid_src=f"{SG_DQG_SQUAD_DATA}/text-data/{dev_or_test}.src.txt",
        # valid_tgt=f"{SG_DQG_SQUAD_DATA}/text-data/{dev_or_test}.tgt.txt",
        # train_ans=f"{SG_DQG_DIR}/datasets/text-data/train.ans.txt",
        # valid_ans=f"{SG_DQG_SQUAD_DATA}/text-data/{dev_or_test}.ans.txt",
        # train_graph=f"{SG_DQG_DIR}/datasets/json-data/train.dp.tag.json",
        # valid_graph=f"{SG_DQG_SQUAD_DATA}/json-data/{dev_or_test}.dp.tag.json",
        train_src=f"{SG_DQG_SQUAD_DATA}/text-data/train.src.txt",
        train_tgt=f"{SG_DQG_SQUAD_DATA}/text-data/train.tgt.txt",
        valid_src=f"{SG_DQG_SQUAD_DATA}/text-data/{dev_or_test}.src.txt",
        valid_tgt=f"{SG_DQG_SQUAD_DATA}/text-data/{dev_or_test}.tgt.txt",
        train_ans=f"{SG_DQG_SQUAD_DATA}/text-data/train.ans.txt",
        valid_ans=f"{SG_DQG_SQUAD_DATA}/text-data/{dev_or_test}.ans.txt",
        train_graph=f"{SG_DQG_SQUAD_DATA}/json-data/train.dp.tag.json",
        valid_graph=f"{SG_DQG_SQUAD_DATA}/json-data/{dev_or_test}.dp.tag.json",
        node_features=True,
        copy=True,
        answer=True,
        feature=False,
        save_sequence_data=f"{sg_dqg_preprocessed_dir}/preprocessed_sequence_data.pt",
        save_graph_data=f"{sg_dqg_preprocessed_dir}/preprocessed_graph_data.pt",
        train_dataset=f"{sg_dqg_datasets_dir}/train_dataset.pt",
        valid_dataset=f"{sg_dqg_datasets_dir}/{dev_or_test}_dataset.pt",
        src_seq_length=200,
        tgt_seq_length=50,
        src_vocab_size=50000,
        tgt_vocab_size=50000,
        src_words_min_frequency=3,
        tgt_words_min_frequency=2,
        vocab_trunc_mode="frequency",
        pre_trained_vocab=GLOVE_PATH,
        word_vec_size=300,
        batch_size=32,
        node_feature=True,
        share_vocab=False,
        pretrained='',
        feat_vocab_size=1000,
        feat_words_min_frequency=1
    )
    create_sg_dqg_dataset(preprocess_args)


def preprocess(dataset_name, graph_type="dp"):
    if dataset_name == "squad":
        if graph_type == "dp":
            if True:
                nlp = spacy.load("en_core_web_sm")
                json_save_dir = f"{SG_DQG_SQUAD_DATA}/json-data"
                text_data_save_dir = f"{SG_DQG_SQUAD_DATA}/text-data"
                Path(json_save_dir).mkdir(parents=True, exist_ok=True)
                Path(text_data_save_dir).mkdir(parents=False, exist_ok=True)

                ds = {'train': SGDQGDataset(
                    dataset_name=dataset_name, data_limit=-1, mode='train', spacy_pipeline=nlp
                ).get_dataset()}
                dev_data = SGDQGDataset(
                    dataset_name=dataset_name, data_limit=-1, mode='dev', spacy_pipeline=nlp
                ).get_split(0.5)
                ds['dev'] = dev_data[:3]
                ds['test'] = dev_data[3:]

                def save_txt_data(d: np.ndarray, filename):
                    np.savetxt(f"{text_data_save_dir}/{filename}", d, delimiter='\n', comments=None, fmt="%s")

                for data_type in ("train", "dev", "test"):
                    deps_path = f"{json_save_dir}/{data_type}.dependencies.graph.json"
                    crefs_path = f"{json_save_dir}/{data_type}.coreferences.graph.json"
                    data_path = f"{json_save_dir}/{data_type}.data.json"
                    contexts, answers, questions = ds[data_type]

                    coref_resolved_context_tokens, ans_indicators = coreference_resolution(
                        contexts, answers
                    )
                    cr_contexts = []
                    corefs = []

                    for context_tokens in coref_resolved_context_tokens:
                        sentences = nlp(" ".join(context_tokens)).sents
                        sentences = [s for s in sentences]
                        cr_contexts.append([evidence.string.strip() for evidence in sentences])
                        corefs.append([[token.text for token in evidence] for evidence in sentences])

                    dependencies = dependency_parsing(cr_contexts, ans_indicators)
                    json_dump(dependencies, deps_path)
                    json_dump(corefs, crefs_path)

                    evidences_list = list([{
                        "index": [paragraph_ind, [ev_ind, ev_ind + 1]],
                        # Ind is the document index, [0,1] corresponds to the evidence span (I think)
                        "text": evidence,
                    } for ev_ind, evidence in enumerate(context)] for paragraph_ind, context in enumerate(cr_contexts))

                    data = list({
                        "question": questions[i],
                        "answer": answers[i],
                        "evidence": evidences_list[i]
                    } for i in range(len(evidences_list)))
                    json_dump(data, data_path)
                    save_txt_data([" ".join(evidences) for evidences in cr_contexts], f"{data_type}.src.txt")
                    save_txt_data([d["question"] for d in data], f"{data_type}.tgt.txt")
                    save_txt_data([d["answer"] for d in data], f"{data_type}.ans.txt")
                    merged_result_path = f"{json_save_dir}/{data_type}.merged.json"
                    merge_dps_and_crefs(data_path, deps_path, crefs_path, merged_result_path)
                    graph_result_path = f"{json_save_dir}/{data_type}.dp.tag.json"
                    build_semantic_graph(merged_result_path, questions, graph_result_path)

            generate_sg_dqg_datasets()
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()


def train(ds_name):
    if ds_name == "squad":
        for training_mode in ("classify", "generate"):
            checkpoint = '' if training_mode == "classify" else f"{SG_DQG_DIR}/models/hotpotqa/classifier_best.chkpt"
            opt = argparse.Namespace(
                sequence_data=f"{SG_DQG_SQUAD_DATA}/preprocessed-data/preprocessed_sequence_data.pt",
                graph_data=f"{SG_DQG_SQUAD_DATA}/preprocessed-data/preprocessed_graph_data.pt",
                train_dataset=f"{SG_DQG_SQUAD_DATA}/Datasets/train_dataset.pt",
                valid_dataset=f"{SG_DQG_SQUAD_DATA}/Datasets/dev_dataset.pt",
                checkpoint=checkpoint,
                epoch=20,
                batch_size=32,
                eval_batch_size=16,
                pre_trained_vocab=True,
                training_mode=training_mode,
                max_token_src_len=200,
                max_token_tgt_len=50,
                sparse=0,
                copy=True,
                coverage=True,
                coverage_weight=0.4,
                node_feature=True,
                d_word_vec=300,
                d_seq_enc_model=512,
                d_graph_enc_model=256,
                n_graph_enc_layer=3,
                d_k=64,
                brnn=True,
                enc_rnn="gru",
                d_dec_model=512,
                n_dec_layer=1,
                dec_rnn="gru",
                maxout_pool_size=2,
                n_warmup_steps=10000,
                dropout=0.5,
                attn_dropout=0.1,
                gpus=[0],
                cuda_seed=-1,
                save_mode="best",
                save_model=f"{SG_DQG_DIR}/models/squad/{training_mode}",
                log_home=f"{SG_DQG_DIR}/logs",
                logfile_train=f"{SG_DQG_DIR}/logs/train_{training_mode}",
                logfile_dev=f"{SG_DQG_DIR}/logs/dev_{training_mode }",
                translate_ppl=15,
                curriculum=0,
                extra_shuffle=True,
                optim="adam",
                learning_rate=0.0002,
                learning_rate_decay=0.75,
                valid_steps=500,
                decay_steps=500,
                start_decay_steps=5000,
                decay_bad_cnt=5,
                max_grad_norm=5,
                max_weight_value=32,
                seed=0,
                pretrained='',
                answer=True,
                feature=False,
                n_seq_enc_layer=1,
                slf_attn=False,
                d_feat_vec=32,
                alpha=0.1,
                layer_attn=False,
                dec_feature=0,
                input_feed=True,
                proj_share_weight=False,
                decay_method=''
            )
            run(opt)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    parser = argparse.ArgumentParser()
    parser.add_argument("action", default="translate", type=str, help='What to do (e.g. "translate")',
                        choices=("translate", "preprocess", "train"))
    parser.add_argument("-ds", help="Which dataset to use for the specified action (e.g \"squad\").", type=str,
                        choices=("squad", "hotpot"), default="squad")

    args = parser.parse_args()
    if args.action == "translate":
        translate(args.ds)
    elif args.action == "preprocess":
        preprocess(args.ds)
    elif args.action == "train":
        train(args.ds)
