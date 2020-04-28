#process SQuAD
#require:CoreNLP
#java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000

import os
import sys
sys.path.append("../")
import json
from tqdm import tqdm
from nltk.tokenize import word_tokenize,sent_tokenize
import collections
import random
from onmt.utils.corenlp import CoreNLP

#remove duplicate words: I love the the bread.
#連続して出てくる単語:the the ...のようなものを取り除く　
#tokenizeもこの中で行う
def overlap_rm(sentence):
    not_rm_list=["(",")"]
    if sentence=="": return ""
    sentence=tokenize(sentence)
    rm_index=[]
    for i in range(len(sentence)-1):
        if sentence[i+1]==sentence[i] and sentence[i+1] not in not_rm_list:
            print(" ".join(sentence))
            print(sentence[i+1])
            print()
            rm_index.append(i+1)
    new_sentence=[sentence[i] for i in range(len(sentence)) if i not in rm_index]
    return " ".join(new_sentence)


def answer_find(context_text,context_sentences,answer_start,answer_end):
    for i,(start_p,end_p) in enumerate(context_sentences):
        #print(start_p,end_p,i)
        if start_p<=answer_start<=end_p:
            sentence_start_pointer=start_p
            sentence_start_id=i
        if start_p<=answer_end<=end_p:
            sentence_end_pointer=end_p
            sentence_end_id=i

    #print(answer_start,answer_end)
    #print()
    answer_text=context_text[context_sentences[sentence_start_id][0]:context_sentences[sentence_end_id][1]]
    pre_text=context_text[context_sentences[sentence_start_id-1][0]:context_sentences[sentence_start_id-1][1]] \
                if sentence_start_id!=0 else ""
    be_text=context_text[context_sentences[sentence_end_id+1][0]:context_sentences[sentence_end_id+1][1]] \
                if sentence_end_id+1!=len(context_sentences) else ""
    return answer_text,pre_text,be_text

#sentenceを受け取り、tokenizeして返す
def tokenize(sent):
    return [token.replace('``','"').replace("''",'"') for token in word_tokenize(sent)]

def data_process(input_path,train=False):
    with open(input_path,"r") as f:
        data=json.load(f)

    corenlp=CoreNLP()
    data_list=[]

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
            context_sentences=corenlp.sentence_tokenize(context_text)
            for qas in paragraph["qas"]:
                all_count+=1
                question_text=qas["question"].lower()
                a=qas["answers"][0]
                answer_text=a["text"].lower()
                answer_start=a["answer_start"]
                answer_end=a["answer_start"]+len(a["text"])

                #contextの中からanswerが含まれる文を見つけ出す
                #対象の文の前後の文も抽出する
                sentence_text,pre_text,be_text=answer_find(context_text,context_sentences,answer_start,answer_end)

                if 0:
                    context_text=overlap_rm(context_text)
                    sentence_text=overlap_rm(sentence_text)
                    pre_text=overlap_rm(pre_text)
                    be_text=overlap_rm(be_text)
                    question_text=overlap_rm(question_text)
                    answer_text=overlap_rm(answer_text)

                interro,non_interro=corenlp.forward(question_text)
                if len(interro)==0:
                    interro=""
                elif interro[-1]=="?":
                    print("interro")
                    interro=" ".join(interro[:-1])
                else:
                    interro=" ".join(interro)
                non_interro=" ".join(non_interro)

                data_list.append({"interro":interro,
                                    "non_interro":non_interro,
                                    "context_sentences":context_sentences,
                                    "sentence_text":sentence_text,
                                    "pre_text":pre_text,
                                    "be_text":be_text,
                                    "question_text":question_text,
                                    "answer_text":answer_text
                                    })
    if train==True:
        with open("data/squad-data-train.json","w")as f:
            json.dump(data_list,f,indent=4)

    if train==False:
        with open("data/squad-data-dev.json","w")as f:
            json.dump(data_list,f,indent=4)

if __name__ == "__main__":
    #main
    random.seed(0)

    data_process(input_path="data/squad-dev-v1.1.json",
                train=False
                )
    data_process(input_path="data/squad-train-v1.1.json",
                train=True
                )
