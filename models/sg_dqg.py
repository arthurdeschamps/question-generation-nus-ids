import argparse
import codecs
import json
import os
import subprocess
import sys
from logging import warning, info
from pathlib import Path
from allennlp.predictors.predictor import Predictor
import allennlp_models.coref # Required
import allennlp_models.syntax.biaffine_dependency_parser # Required
from nltk.tree import Tree

from tqdm import tqdm

from data_processing.nqg_dataset import NQGDataset
from defs import SG_DQG_DATA, SG_DQG_DIR, RESULTS_DIR

sys.path.append(f"{SG_DQG_DIR}/build-semantic-graphs")
sys.path.append(f"{SG_DQG_DIR}/src")

from translate import main as sg_dqg_translate
from merge import merge
from build_semantic_graph import run as build_semantic_graph

json_load = lambda x: json.load(codecs.open(x, 'r', encoding='utf-8'))
json_dump = lambda d, p: json.dump(d, codecs.open(p, 'w', 'utf-8'), indent=2, ensure_ascii=False)


def translate(dataset_name):
    if dataset_name == "squad":
        data_home = f"{SG_DQG_DIR}/datasets"
        log_home = f"{RESULTS_DIR}/sg_dqg"

        class OptTranslate:
            model = f"{SG_DQG_DIR}/models/generator_best.chkpt"
            sequence_data = f"{data_home}/preprocessed-data/preprocessed_sequence_data.pt"
            graph_data = f"{data_home}/preprocessed-data/preprocessed_graph_data.pt"
            valid_data = f"{data_home}/Datasets/valid_dataset.pt"
            output = f"{log_home}/prediction.txt"
            beam_size = 5
            batch_size = 16
            gpus = [0]
            cuda = True

        sg_dqg_translate(OptTranslate())

    else:
        raise NotImplementedError()


def dependency_parsing(evidences_list):
    predictor = Predictor.from_path(
        "https://storage.googleapis.com/allennlp-public-models/biaffine-dependency-parser-ptb-2020.04.06.tar.gz",
        cuda_device=0
    )
    dps = []
    info("Performing dependency parsing...")
    for evidences in tqdm(evidences_list):
        dp = []
        for evidence in evidences:
            pred = predictor.predict(
                sentence=str(evidence['text'])
            )
            nodes = []
            for i in range(len(pred['words'])):
                nodes.append({
                    'head': pred['predicted_heads'][i] - 1,
                    'pos': pred['pos'][i],
                    'dep': pred['predicted_dependencies'][i],
                    'word': pred['words'][i]
                })
            dp.append(nodes)
        dps.append(dp)
    return dps


def coreference_resolution(evidences_list):
    predictor = Predictor.from_path(
        "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz",
        cuda_device=0
    )
    coreferences_list = []
    kept_indices = []
    info("Performing coreference resolutions...")
    for i, evidences in tqdm(enumerate(evidences_list)):
        cr_solved_words = []
        for evidence in evidences:
            try:
                cr = predictor.predict(
                    document=str(evidence['text'])
                )
                words = cr['document']
                for span_index, span in enumerate(cr['top_spans']):
                    # Pronouns have only one word
                    if span[1] == span[0] and cr['predicted_antecedents'][span_index] > -1:
                        antecedent_span = cr['top_spans'][cr['predicted_antecedents'][span_index]]
                        left = antecedent_span[0]
                        right = antecedent_span[1]
                        words = words[:span[0]] + words[left:right+1] + words[span[0]+1:]
                cr_solved_words.append(words)
                kept_indices.append(i)
            except TypeError:
                warning(f"Couldn't run coreference resolution on:\n\"{evidence}\"")
                cr_solved_words.append([])
        coreferences_list.append(cr_solved_words)
    return kept_indices, coreferences_list


def merge_dps_and_crefs(data_path, dependencies_path, coreferences_path, result_path):
    data = merge(json_load(data_path), json_load(dependencies_path), json_load(coreferences_path))
    json_dump(data, result_path)


def preprocess(dataset_name, graph_type="dp"):
    if dataset_name == "squad":
        if graph_type == "dp":
            save_dir = f"{SG_DQG_DATA}/squad/json-data"
            if not (os.path.isdir(save_dir)):
                Path(save_dir).mkdir(parents=True, exist_ok=True)

            ds = {'train': NQGDataset(
                dataset_name=dataset_name, data_limit=1, break_up_paragraphs=True, mode='train'
            ).get_dataset()}
            dev_data = NQGDataset(
                dataset_name=dataset_name, data_limit=1, break_up_paragraphs=True, mode='dev'
            ).get_split(0.5)
            ds['dev'] = dev_data[:3]
            ds['test'] = dev_data[3:]

            for data_type in ("train", "dev", "test"):
                deps_path = f"{save_dir}/{data_type}.dependencies.graph.json"
                crefs_path = f"{save_dir}/{data_type}.coreferences.graph.json"
                data_path = f"{save_dir}/{data_type}.data.json"
                contexts, answers, questions = ds[data_type]
                evidences_list = list([{
                    "index": [ind, [0, 1]],
                    # Ind is the document index, [0,1] corresponds to the evidence span (I think)
                    "text": context.text
                }] for ind, context in enumerate(contexts))

                data = list({
                    "question": question,
                    "answer": answer.text,
                    "evidence": evidences
                } for question, answer, evidences in zip(questions, answers, evidences_list))
                kept_indices, coreferences = coreference_resolution(evidences_list)
                dependencies = dependency_parsing(evidences_list)
                json_dump(dependencies, deps_path)
                json_dump(coreferences, crefs_path)
                json_dump(list(data[i] for i in kept_indices), data_path)

                merged_result_path = f"{save_dir}/{data_type}.merged.json"
                merge_dps_and_crefs(data_path, deps_path, crefs_path, merged_result_path)
                graph_result_path = f"{save_dir}/{data_type}.dp.tag.json"
                build_semantic_graph(merged_result_path, questions, graph_result_path)
        else:
            raise NotImplementedError()
        exit()
        subprocess.run(
            f"cd",
            SG_DQG_DIR
        )
        subprocess.run(
            "python preprocess.py"
        )
    else:
        raise NotImplementedError()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("action", default="translate", type=str, help='What to do (e.g. "translate")',
                        choices=("translate", "preprocess"))
    parser.add_argument("ds", help="Which dataset to use for the specified action (e.g \"squad\").", type=str,
                        choices=("squad",))

    args = parser.parse_args()
    if args.action == "translate":
        translate(args.ds)
    elif args.action == "preprocess":
        preprocess(args.ds)
