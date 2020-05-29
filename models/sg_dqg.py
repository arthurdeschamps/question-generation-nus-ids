import argparse
import codecs
import json
import sys
from logging import warning, info, debug
from pathlib import Path
from allennlp.predictors.predictor import Predictor
import allennlp_models.coref  # Required
import allennlp_models.syntax.biaffine_dependency_parser  # Required
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


def json_load(x):
    return json.load(codecs.open(x, 'r', encoding='utf-8'))


def json_dump(d, p):
    return json.dump(d, codecs.open(p, 'w', 'utf-8'), indent=2, ensure_ascii=False)


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
    for ev_index, evidences in tqdm(enumerate(resolved_corefs)):
        dp = []
        for evidence in evidences:
            pred = predictor.predict(
                sentence=evidence
            )

            nodes = []
            for i in range(len(pred['words'])):
                nodes.append({
                    'head': pred['predicted_heads'][i] - 1,
                    'pos': pred['pos'][i],
                    'dep': pred['predicted_dependencies'][i],
                    'word': pred['words'][i],
                    'ans': updated_bios[ev_index][i]
                })
            dp.append(nodes)
        dps.append(dp)
    return dps


def coreference_resolution(evidences_list, answers):
    ans_indicators = []
    predictor = Predictor.from_path(
        "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz",
        cuda_device=0
    )
    coreferences_list = []
    info("Performing coreference resolutions...")
    for i, (evidences, answer) in tqdm(enumerate(zip(evidences_list, answers))):
        cr_solved_words = []
        for evidence in evidences:
            cr = predictor.predict(
                document=str(evidence['text'])
            )
            words = cr['document']
            substitutions = {}
            start_index, end_index = answer_span(words, answer.split(' '))
            if start_index is None or end_index is None:
                warning(f"Error with answer \"{answer}\"")
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
            cr_solved_words.append(resolved)
        coreferences_list.append(cr_solved_words)
    return coreferences_list, ans_indicators


def merge_dps_and_crefs(data_path, dependencies_path, coreferences_path, result_path):
    data = merge(json_load(data_path), json_load(dependencies_path), json_load(coreferences_path))
    json_dump(data, result_path)


def generate_sg_dqg_datasets(dev_or_test="dev"):
    sg_dqg_preprocessed_dir = f"{SG_DQG_SQUAD_DATA}/preprocessed-data/"
    sg_dqg_datasets_dir = f"{SG_DQG_SQUAD_DATA}/Datasets"
    Path(sg_dqg_preprocessed_dir).mkdir(parents=False, exist_ok=True)
    Path(sg_dqg_datasets_dir).mkdir(parents=False, exist_ok=True)

    preprocess_args = argparse.Namespace(
        train_src=f"{SG_DQG_DIR}/datasets/text-data/train.src.txt",
        train_tgt=f"{SG_DQG_DIR}/datasets/text-data/train.tgt.txt",
        valid_src=f"{SG_DQG_SQUAD_DATA}/text-data/{dev_or_test}.src.txt",
        valid_tgt=f"{SG_DQG_SQUAD_DATA}/text-data/{dev_or_test}.tgt.txt",
        train_ans=f"{SG_DQG_DIR}/datasets/text-data/train.ans.txt",
        valid_ans=f"{SG_DQG_SQUAD_DATA}/text-data/{dev_or_test}.ans.txt",
        train_graph=f"{SG_DQG_DIR}/datasets/json-data/train.dp.tag.json",
        valid_graph=f"{SG_DQG_SQUAD_DATA}/json-data/{dev_or_test}.dp.tag.json",
        # train_src=f"{SG_DQG_SQUAD_DATA}/text-data/train.src.txt",
        # train_tgt=f"{SG_DQG_SQUAD_DATA}/text-data/train.tgt.txt",
        # valid_src=f"{SG_DQG_SQUAD_DATA}/text-data/{dev_or_test}.src.txt",
        # valid_tgt=f"{SG_DQG_SQUAD_DATA}/text-data/{dev_or_test}.tgt.txt",
        # train_ans=f"{SG_DQG_SQUAD_DATA}/text-data/train.ans.txt",
        # valid_ans=f"{SG_DQG_SQUAD_DATA}/text-data/{dev_or_test}.ans.txt",
        # train_graph=f"{SG_DQG_SQUAD_DATA}/json-data/train.dp.tag.json",
        # valid_graph=f"{SG_DQG_SQUAD_DATA}/json-data/{dev_or_test}.dp.tag.json",
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
                json_save_dir = f"{SG_DQG_SQUAD_DATA}/json-data"
                text_data_save_dir = f"{SG_DQG_SQUAD_DATA}/text-data"
                Path(json_save_dir).mkdir(parents=True, exist_ok=True)
                Path(text_data_save_dir).mkdir(parents=False, exist_ok=True)

                ds = {'train': SGDQGDataset(
                    dataset_name=dataset_name, data_limit=5, mode='train'
                ).get_dataset()}
                dev_data = SGDQGDataset(
                    dataset_name=dataset_name, data_limit=None, mode='dev'
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
                    evidences_list = list([{
                        "index": [ind, [0, 1]],
                        # Ind is the document index, [0,1] corresponds to the evidence span (I think)
                        "text": context,
                    }] for ind, context in enumerate(contexts))

                    coref_resolved_context_tokens, ans_indicators = coreference_resolution(
                        evidences_list, answers
                    )
                    cr_contexts = [[" ".join(context) for context in contexts] for contexts in
                                   coref_resolved_context_tokens]

                    dependencies = dependency_parsing(cr_contexts, ans_indicators)
                    json_dump(dependencies, deps_path)
                    json_dump(coref_resolved_context_tokens, crefs_path)

                    sources = []
                    for cr_evidences, evidences in zip(cr_contexts, evidences_list):
                        for cr_evidence, evidence in zip(cr_evidences, evidences):
                            evidence["text"] = cr_evidence
                            sources.append(cr_evidence)

                    data = list({
                        "question": questions[i],
                        "answer": answers[i],
                        "evidence": evidences_list[i]
                    } for i in range(len(evidences_list)))
                    json_dump(data, data_path)
                    save_txt_data(sources, f"{data_type}.src.txt")
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
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("action", default="translate", type=str, help='What to do (e.g. "translate")',
                        choices=("translate", "preprocess"))
    parser.add_argument("-ds", help="Which dataset to use for the specified action (e.g \"squad\").", type=str,
                        choices=("squad", "hotpot"), default="squad")

    args = parser.parse_args()
    if args.action == "translate":
        translate(args.ds)
    elif args.action == "preprocess":
        preprocess(args.ds)
    elif args.action == "train":
        train(args.ds)
