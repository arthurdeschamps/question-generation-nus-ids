#SQuADのデータ処理
#必要条件:CoreNLP
#Tools/core...で
#java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000

import os
import sys
sys.path.append("../")
import json
import gzip
import pandas as pd
import numpy as np
from tqdm import tqdm
from nltk.tokenize import word_tokenize,sent_tokenize
import pickle
import collections
import random
from onmt.utils.corenlp import CoreNLP

def answer_find(context_text,answer_start,answer_end):
    context=sent_tokenize(context_text)
    start_p=0

    #start_p:対象となる文の文字レベルでの始まりの位置
    #end_p:対象となる文の文字レベルでの終端の位置
    #answer_startがstart_pからend_pの間にあるかを確認。answer_endも同様
    for i,sentence in enumerate(context):
        end_p=start_p+len(sentence)
        if start_p<=answer_start and answer_start<=end_p:
            sentence_start_id=i
        if start_p<=answer_end and answer_end<=end_p:
            sentence_end_id=i
        #スペースが消えている分の追加、end_pの計算のところでするべきかは不明
        start_p+=len(sentence)+1
    #得られた文を結合する（大抵は文は一つ）
    answer_sentence=" ".join(context[sentence_start_id:sentence_end_id+1])

    return answer_sentence

#sentenceを受け取り、tokenizeして返す
def tokenize(sent):
    return [token.replace('``','"').replace("''",'"') for token in word_tokenize(sent)]

def data_process(input_path,train=False):
    with open(input_path,"r") as f:
        data=json.load(f)

    corenlp=CoreNLP()
    interro_list=[]

    questions=[]
    answers=[]
    sentences=[]

    all_count=0

    #context_text:文章
    #question_text:質問
    #answer_text:解答
    #answer_start,answer_end:解答の文章の中での最初と最後の位置
    for topic in tqdm(data["data"]):
        topic=topic["paragraphs"]
        for paragraph in topic:
            context_text=paragraph["context"].lower()
            for qas in paragraph["qas"]:
                all_count+=1
                question_text=qas["question"].lower()
                a=qas["answers"][0]
                answer_text=a["text"].lower()
                answer_start=a["answer_start"]
                answer_end=a["answer_start"]+len(a["text"])

                #contextの中からanswerが含まれる文を見つけ出す
                sentence_text=answer_find(context_text,answer_start,answer_end)

                question_text=" ".join(tokenize(question_text))
                sentence_text=" ".join(tokenize(sentence_text))
                answer_text=" ".join(tokenize(answer_text))

                interro,non_interro=corenlp.forward(question_text)
                if len(interro)==0:
                    interro=""
                elif "?" in interro:
                    print("interro")
                    interro=" ".join(interro)
                else:
                    interro=" ".join(interro+["?"])
                non_interro=" ".join(non_interro)

                interro_list.append({"interro":interro,"non_interro":non_interro})



    if train==True:
        with open("data/squad-interro-train","w")as f:
            json.dump(interro_list,f)

    if train==False:
        with open("data/squad-interro-dev","w")as f:
            json.dump(interro_list,f)



if __name__ == "__main__":
    #main
    random.seed(0)

    data_process(input_path="data/squad-dev-v1.1.json",
                train=False
                )

    data_process(input_path="data/squad-train-v1.1.json",
                train=True
                )
