import sys
from tqdm import tqdm

from build_tree import build_tree
from prune_and_merge_tree import prune
from rearrange_tree import rearrange
from build_graph import get_graph
from merge_graph import merge
from tag import text_load, main

import json
import codecs

json_load = lambda x: json.load(codecs.open(x, 'r', encoding='utf-8'))
json_dump = lambda d, p: json.dump(d, codecs.open(p, 'w', 'utf-8'), indent=2, ensure_ascii=False)


def run(data_path, questions, graph_save_path):
    data = json_load(data_path)

    graphs = []
    for idx, sample in tqdm(enumerate(data), desc='   - (Building Graphs) -   '):
        corpus = sample['evidence']
        evidence = []
        for sent in corpus:
            sent = build_tree(sent)
            sent = {'sequence': sent['words'], 'tree': prune(sent['tree'], sent['words'])}
            sent = {'sequence': sent['sequence'], 'tree': rearrange(sent['tree'], sent['sequence'])}
            evidence.append({'sequence': sent['sequence'], 'graph': get_graph(sent['tree'])})
        graph = merge(evidence)
        graphs.append(graph)

    graphs = main(graphs, questions)

    json_dump(graphs, graph_save_path)


if __name__ == '__main__':
    questions = text_load(sys.argv[2])
    run(sys.argv[1], questions, sys.argv[3])
