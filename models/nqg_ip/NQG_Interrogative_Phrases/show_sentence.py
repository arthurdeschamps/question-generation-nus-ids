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
import numpy as np




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
parser.add_argument("--result", action="store_true")
parser.add_argument("--show", action="store_true")
parser.add_argument("--size", type=int,default="110")

args = parser.parse_args()

random.seed(0)



srcs=[]
tgts=[]
preds=[]



np.random.seed(0)
id_list=np.random.permutation(list(range(4658)))
id_shuffle_list=[np.random.permutation(list(range(2))) for i in range(args.size)]


if args.show:
    with open("data/squad-src-test-interro-answer.txt","r")as f:
        for line in f:
            srcs.append(line.strip())

    with open("data/squad-tgt-test-interro.txt","r")as f:
        for line in f:
            tgts.append(line.strip())

    if 0:
        pred_name=["data/squad-pred-test-interro.txt",
                    "data/squad-pred-test-nqg.txt",
                    "data/squad-pred-test-repanswer.txt",
                    "data/squad-pred-test-interro-repanswer.txt"]
    if 1:
        pred_name=["data/squad-pred-test-repanswer.txt",
                    "data/squad-pred-test-interro-repanswer.txt"]

    for name in pred_name:
        mylist=[]
        with open(name,"r")as f:
            for line in f:
                mylist.append(line.strip())
        preds.append(mylist)

    for i,id in enumerate(id_list[0:args.size]):
        print("problem:{}".format(i))
        print("SRC:{}".format(srcs[id]))
        print("TGT:{}".format(tgts[id]))

        for j in id_shuffle_list[i]:
            print(preds[j][id])
        print()


else:
    data=[]
    with open("human_eval_2.txt","r")as f:
        for line in f:
            data.append(line.rstrip())

    score=np.zeros((110,2,2)) #->100,2,2
    count=0
    num_list=["0","1","2"]
    for i,line in enumerate(data):
        if line[-3:]==" NG":
            count+=2
        elif len(line)==3 and line[0] in num_list and line[2] in num_list:
            if id_shuffle_list[int(count/2)][count%2]==1: print(count/2,line)
            score[int(count/2)][id_shuffle_list[int(count/2)][count%2]]=[int(l) for l in line.split()]
            count+=1
            #score.append(int(line[0])-int(line[2]))
            #score.append(int(line[0]))
    #print("result of repanswer:{}".format(np.average(score[:,0])))
    #print("result of interro-repanswer:{}".format(np.average(score[:,1])))


    #print(score)
    #print(count)

    print(":{}".format(sum([1 if s in [2] else 0 for s in score[:,0,0]])))
    print(":{}".format(sum([1 if s in [1] else 0 for s in score[:,0,0]])))
    print(":{}".format(sum([1 if s in [0] else 0 for s in score[:,0,0]])))

    print(":{}".format(sum([1 if s in [2] else 0 for s in score[:,1,0]])))
    print(":{}".format(sum([1 if s in [1] else 0 for s in score[:,1,0]])))
    print(":{}".format(sum([1 if s in [0] else 0 for s in score[:,1,0]])))

    print(":{}".format(sum([1 if s in [2] else 0 for s in score[:,0,1]])))
    print(":{}".format(sum([1 if s in [1] else 0 for s in score[:,0,1]])))
    print(":{}".format(sum([1 if s in [0] else 0 for s in score[:,0,1]])))

    print(":{}".format(sum([1 if s in [2] else 0 for s in score[:,1,1]])))
    print(":{}".format(sum([1 if s in [1] else 0 for s in score[:,1,1]])))
    print(":{}".format(sum([1 if s in [0] else 0 for s in score[:,1,1]])))
