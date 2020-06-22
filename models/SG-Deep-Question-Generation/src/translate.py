import math
import torch
from torch import cuda
import torch.nn as nn
import argparse
from tqdm import tqdm
from numpy import loadtxt
from onqg.utils.translate import Translator
from onqg.dataset import Dataset
from onqg.utils.model_builder import build_model


def dump(data, filename):
    golds, preds, paras = data[0], data[1], data[2]
    with open(filename, 'w', encoding='utf-8') as f:
        for g, p, pa in zip(golds, preds, paras):
            pa = [w for w in pa if w not in ['[PAD]', '[CLS]']]
            f.write('<para>\t' + ' '.join(pa) + '\n')
            f.write('<gold>\t' + ' '.join(g[0]) + '\n')
            f.write('<pred>\t' + ' '.join(p) + '\n')
            f.write('===========================\n')


def main(opt):
    if opt.cuda:
        cuda.set_device(opt.gpus[0])
    device = torch.device('cuda' if opt.cuda else 'cpu')

    checkpoint = torch.load(opt.model)
    model_opt = checkpoint['settings']
    model_opt.gpus = opt.gpus
    model_opt.beam_size, model_opt.batch_size, model_opt.n_best = opt.beam_size, opt.batch_size, opt.n_best

    ### Prepare Data ###
    sequences = torch.load(opt.sequence_data)
    seq_vocabularies = sequences['dict']

    if hasattr(opt, "valid_data"):
        validData = torch.load(opt.valid_data)
    else:
        validData = loadtxt(opt.valid_data_txt, delimiter='\n', comments=None, dtype=str)
        validData = torch.tensor(validData, requires_grad=False)
    validData.batchSize = opt.batch_size
    validData.numBatches = math.ceil(len(validData.src) / validData.batchSize)

    ### Prepare Model ###
    validData.device = validData.device = device 
    model, _ = build_model(model_opt, device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    translator = Translator(model_opt, seq_vocabularies['tgt'], sequences['valid']['tokens'], seq_vocabularies['src'])

    bleu, outputs = translator.eval_all(model, validData, output_sent=True)

    print('\nbleu-4', bleu, '\n')

    dump(outputs, opt.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='translate.py')

    parser.add_argument('-model', required=True, help='Path to model .pt file')
    parser.add_argument('-sequence_data', required=True, help='Path to data file')
    parser.add_argument('-graph_data', required=True, help='Path to data file')
    parser.add_argument('-valid_data', required=True, help='Path to data file')
    parser.add_argument('-output', required=True, help='Path to output the predictions')
    parser.add_argument('-beam_size', type=int, default=5)
    parser.add_argument('-batch_size', type=int, default=32)
    parser.add_argument('-gpus', default=[], nargs='+', type=int)

    opt = parser.parse_args()
    opt.cuda = True if opt.gpus else False
    
    main(opt)