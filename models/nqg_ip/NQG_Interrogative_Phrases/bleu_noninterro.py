import warnings
warnings.filterwarnings("ignore")
import nltk
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.translate.bleu_score import corpus_bleu,sentence_bleu
from onmt.utils.corenlp import CoreNLP
import numpy as np

from tqdm import tqdm
from collections import defaultdict
import random
import json
import argparse
import os.path

parser = argparse.ArgumentParser()
parser.add_argument("--src", type=str, default="data/squad-src-val-interro.txt", help="input model epoch")
parser.add_argument("--tgt", type=str, default="data/squad-tgt-val-interro.txt", help="input model epoch")
parser.add_argument("--pred", type=str, default="pred.txt", help="input model epoch")
parser.add_argument("--interro", type=str, default="data/squad-interro-val-interro.txt", help="input model epoch")
parser.add_argument("--noninterro", type=str, default="data/squad-noninterro-val-interro.txt", help="input model epoch")
parser.add_argument("--p_noninterro", type=str, default="data/squad-pred_noninterro-val-interro.txt", help="input model epoch")

parser.add_argument("--tgt_interro", type=str, default="", help="input model epoch")
parser.add_argument("--not_interro", action="store_true")
parser.add_argument("--same_interro", action="store_true")
parser.add_argument("--each_interro", action="store_true")

parser.add_argument("--ratio", type=float, default=1.0, help="input model epoch")
parser.add_argument("--print", type=int ,default=0)
args = parser.parse_args()

random.seed(0)

srcs=[]
targets=[]
predicts=[]
interros=[]
t_noninterros=[]
p_noninterros=[]

with open(args.src,"r")as f:
    for line in f:
        srcs.append(line.strip())

with open(args.tgt,"r")as f:
    for line in f:
        targets.append(line.strip())

with open(args.pred,"r")as f:
    for line in f:
        predicts.append(line.strip())

with open(args.interro,"r")as f:
    for line in f:
        interros.append(line.strip())

with open(args.noninterro,"r")as f:
    for line in f:
        t_noninterros.append(line.strip().split())

if os.path.exists(args.p_noninterro) and False:
    with open(args.p_noninterro,"r")as f:
        for line in f:
            p_noninterros.append(line.strip().split())
else:
    corenlp=CoreNLP()
    for p in tqdm(predicts):
        interro,p_noninterro=corenlp.forward(p)
        p_noninterros.append(p_noninterro)
    with open(args.p_noninterro,"w")as f:
        for line in p_noninterros:
            f.write(" ".join(line)+"\n")

if args.tgt_interro!="":
    data_size=len(srcs)
    srcs=[srcs[i] for i in range(data_size) if args.tgt_interro=="" or args.tgt_interro in interros[i]]
    targets=[targets[i] for i in range(data_size) if args.tgt_interro=="" or args.tgt_interro in interros[i]]
    predicts=[predicts[i] for i in range(data_size) if args.tgt_interro=="" or args.tgt_interro in interros[i]]
    t_noninterros=[t_noninterros[i] for i in range(data_size) if args.tgt_interro=="" or args.tgt_interro in interros[i]]
    p_noninterros=[p_noninterros[i] for i in range(data_size) if args.tgt_interro=="" or args.tgt_interro in interros[i]]
    print(len(srcs),args.tgt_interro)

np.random.seed(0)
randomlist=np.random.permutation(np.arange(len(srcs)))
for i in randomlist[0:args.print]:
    print(srcs[i])
    print(targets[i])
    print(predicts[i])
    print()

if args.not_interro:
    targets_set=[[t] for t in t_noninterros]
    predicts_set=[p for p in p_noninterros]
elif args.same_interro:
    target_dict=defaultdict(lambda:[])
    predict_dict=defaultdict(str)
    src_set=set(srcs)
    for s,t,p in zip(srcs,t_noninterros,p_noninterros):
        target_dict[s].append(t)
        predict_dict[s]=p
    targets_set=[target_dict[s] for s in src_set if s in target_dict]
    predicts_set=[predict_dict[s] for s in src_set if s in predict_dict]
elif args.each_interro:
    target_dict=defaultdict(lambda:[])
    predict_dict=defaultdict(str)
    src_set=set(srcs)
    for s,t,p in zip(srcs,t_noninterros,p_noninterros):
        target_dict[s].append(t)
        predict_dict[" ".join(p)]=s
    targets_set=[target_dict[predict_dict[" ".join(p)]] for p in p_noninterros]
    predicts_set=p_noninterros


if 0:
    #print(len(targets_set),len(predicts_set))
    print()
    print("this is the bleu score without interrogative phrases")
    print(corpus_bleu(targets_set,predicts_set,weights=(1,0,0,0)))
    print(corpus_bleu(targets_set,predicts_set,weights=(0.5,0.5,0,0)))
    print(corpus_bleu(targets_set,predicts_set,weights=(0.333,0.333,0.333,0)))
    print(corpus_bleu(targets_set,predicts_set,weights=(0.25,0.25,0.25,0.25)))

    print()

    ####normal bleu

    targets=[t.split() for t in targets]
    predicts=[p.split() for p in predicts]
    src_set=set(srcs)

    #それぞれリファレンスは1対１対応
    if args.not_interro:
        targets=[[t] for t in targets]
        predicts=predicts
    #srcの文を使用して同じものは全てreference
    elif args.same_interro:
        target_dict=defaultdict(lambda: [])
        predict_dict=defaultdict(str)
        for s,t,p,i in zip(srcs,targets,predicts,interros):
            target_dict[s].append(t)
            predict_dict[s]=p
        targets=[target_dict[s] for s in src_set if s in target_dict]
        predicts=[predict_dict[s] for s in src_set if s in predict_dict]
    #文と疑問詞が同じもののみをreferenceとして利用する。predictの文は全て違うものであると仮定する
    elif args.each_interro:
        target_dict=defaultdict(lambda:[])
        predict_dict=defaultdict(str)
        for s,t,p in zip(srcs,targets,predicts):
            target_dict[s].append(t)
            predict_dict[" ".join(p)]=s
        targets=[target_dict[predict_dict[" ".join(p)]] for p in predicts]
        predicts=predicts

    print()
    print("this is the normal bleu score")
    print(corpus_bleu(targets,predicts,weights=(1,0,0,0)))
    print(corpus_bleu(targets,predicts,weights=(0.5,0.5,0,0)))
    print(corpus_bleu(targets,predicts,weights=(0.333,0.333,0.333,0)))
    print(corpus_bleu(targets,predicts,weights=(0.25,0.25,0.25,0.25)))
