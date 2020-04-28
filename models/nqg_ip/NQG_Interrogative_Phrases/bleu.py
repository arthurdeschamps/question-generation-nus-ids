import warnings
warnings.filterwarnings("ignore")
import nltk
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.translate.bleu_score import corpus_bleu,sentence_bleu
from tqdm import tqdm
from collections import defaultdict
import collections
import math
from statistics import mean, median,variance,stdev
import random
import json
import argparse




parser = argparse.ArgumentParser()
parser.add_argument("--src", type=str, default="data/squad-src-val-interro.txt", help="input model epoch")
parser.add_argument("--tgt", type=str, default="data/squad-tgt-val-interro.txt", help="input model epoch")
parser.add_argument("--pred", type=str, default="pred.txt", help="input model epoch")
parser.add_argument("--interro", type=str, default="data/squad-interro-val-interro.txt", help="input model epoch")

parser.add_argument("--tgt_interro", type=str, default="", help="input model epoch")
parser.add_argument("--not_interro", action="store_true")
parser.add_argument("--all_interro", action="store_true")
parser.add_argument("--interro_each", action="store_true")

parser.add_argument("--notsplit", action="store_true")
parser.add_argument("--print", action="store_true")


args = parser.parse_args()

random.seed(0)

srcs=[]
targets=[]
predicts=[]
interros=[]

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


#srcs=[s.split() for s in targets]
targets=[t.split() for t in targets]
predicts=[p.split() for p in predicts]

#それぞれリファレンスは1対１対応
if args.not_interro:
    targets=[[t] for t in targets]
    predicts=predicts
#srcの文を使用して同じものは全てreference
elif args.all_interro:
    target_dict=defaultdict(lambda: [])
    predict_dict=defaultdict(str)
    src_set=set(srcs)
    for s,t,p,i in zip(srcs,targets,predicts,interros):
        target_dict[s].append(t)
        predict_dict[s]=p
    targets=[target_dict[s] for s in src_set if s in target_dict]
    predicts=[predict_dict[s] for s in src_set if s in predict_dict]
#文と疑問詞が同じもののみをreferenceとして利用する。predictの文は全て違うものであると仮定する
elif args.interro_each:
    target_dict=defaultdict(lambda:[])
    predict_dict=defaultdict(str)
    src_set=set(srcs)
    for s,t,p in zip(srcs,targets,predicts):
        target_dict[s].append(t)
        predict_dict[" ".join(p)]=s
    targets=[target_dict[predict_dict[" ".join(p)]] for p in predicts]
    predicts=predicts



if args.print:
    for i in range(5):
        print("target:{}".format(" ".join(targets[i][0])))
        print("predict:{}".format(" ".join(predicts[i])))
        print()

print(len(targets),len(predicts))
print(corpus_bleu(targets,predicts,weights=(1,0,0,0)))
print(corpus_bleu(targets,predicts,weights=(0.5,0.5,0,0)))
print(corpus_bleu(targets,predicts,weights=(0.333,0.333,0.333,0)))
print(corpus_bleu(targets,predicts,weights=(0.25,0.25,0.25,0.25)))
