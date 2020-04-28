#SQuADのデータ処理
#必要条件:CoreNLP
#Tools/core...で
#java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000

import os
import sys
sys.path.append("../")
import json
import re
from tqdm import tqdm
from nltk.tokenize import word_tokenize,sent_tokenize
import collections
import random
from nltk.corpus import stopwords
from nltk.corpus import stopwords


#sentenceとquestionで共通するノンストップワードが一つもない場合False
def check_overlap(sentence,question,stop_words):

    for w in question.split():
        if sentence.find(w)!=-1 and w not in stop_words:
            return True
    return False

def answer_find(context_text,answer_start,answer_end):

    #context=sent_tokenize(context_text)
    context=re.split("\.\s|\?\s|\!\s",context_text)
    start_p=0

    for i,sentence in enumerate(context):
        start_p=context_text.find(sentence,start_p)
        end_p=start_p+len(sentence)+1

        if start_p<=answer_start<end_p:
            sentence_start_id=i
        if start_p<answer_end<=end_p:
            sentence_end_id=i
        #スペースが消えている分の追加、end_pの計算のところでするべきかは不明
        #findで処理する
        start_p+=len(sentence)


    #得られた文を結合する（大抵は文は一つ）
    answer_sentence=context_text[sentence_start_id:sentence_end_id+1]
    if sentence_start_id!=sentence_end_id:
        print(context_text)
        print(context)
        print(answer_start,answer_end)
        print(answer_sentence)
        print()


    return answer_sentence

#sentenceを受け取り、tokenizeして返す
def tokenize(sent):
    return [token.replace('``','"').replace("''",'"') for token in word_tokenize(sent)]

#単語が連続して現れている部分は削除する
def overlap_rm(sentence):
    #print(sentence)
    sentence=sentence.split()
    #print(sentence)
    #print()
    rm_index=[]
    for i in range(len(sentence)-1):
        if sentence[i+1]==sentence[i]:
            rm_index.append(i+1)
    new_sentence=[sentence[i] for i in range(len(sentence)) if i not in rm_index]
    return " ".join(new_sentence)


def data_process(input_path,interro_path,train=False):
    with open(input_path,"r") as f:
        data=json.load(f)
    with open(interro_path,"r") as f:
        interro_data=json.load(f)

    use_interro=True
    use_answer=True
    use_pre_interro=False
    replace_answer=True

    questions=[]
    answers=[]
    sentences=[]
    interros=[]
    non_interros=[]
    stop_words = stopwords.words('english')


    all_count=0
    overlap_count=0
    noise_count=0
    non_interro_count=0

    for topic in tqdm(data["data"]):
        topic=topic["paragraphs"]
        for paragraph in topic:
            context_text=paragraph["context"].lower()
            for qas in paragraph["qas"]:
                sentence_text=interro_data[all_count]["sentence_text"]
                question_text=interro_data[all_count]["question_text"]
                answer_text=interro_data[all_count]["answer_text"]
                interro_text=interro_data[all_count]["interro"]
                non_interro_text=interro_data[all_count]["non_interro"]

                answer_start=qas["answers"][0]["answer_start"]

                all_count+=1

                if len(sentence_text)<=5 or len(question_text)<=5:
                    noise_count+=1
                    continue

                #疑問詞がないものは削除
                if interro_text=="":
                    non_interro_count+=1
                    continue

                #テキストとノンストップワードが一つも重複してないものは除去
                if check_overlap(sentence_text,question_text,stop_words)==False:
                    overlap_count+=1
                    continue

                if interro_text[-1]=="?":
                    print(interro_text)
                    interro_text=interro_text[:-2]
                    print(interro_text)

                if replace_answer:
                    sentence_start=context_text.find(sentence_text)
                    answer_start_insent=answer_start-sentence_start
                    answer_end_insent=answer_start_insent+len(answer_text)
                    sentence_text=sentence_text[:answer_start_insent] \
                                    +"answer_hidden_token" \
                                    +sentence_text[answer_end_insent:]

                sentence_text=" ".join(tokenize(sentence_text))
                question_text=" ".join(tokenize(question_text))
                answer_text=" ".join(tokenize(answer_text))
                interro_text=" ".join(tokenize(interro_text))
                non_interro_text=" ".join(tokenize(non_interro_text))

                if use_interro and not use_answer:
                    sentence_text=" ".join([sentence_text,"<SEP>",interro_text])
                elif not use_interro and use_answer:
                    sentence_text=" ".join([sentence_text,"<SEP>",answer_text])
                elif use_interro and use_answer:
                    sentence_text=" ".join([sentence_text,"<SEP>",interro_text,"<SEP2>",answer_text])

                sentences.append(sentence_text)
                questions.append(question_text)
                answers.append(answer_text)
                interros.append(interro_text)
                non_interros.append(non_interro_text)


    print(all_count)

    if replace_answer:
        if use_interro and use_answer:
            setting="-interro-repanswer"
        elif not use_interro and use_answer:
            setting="-repanswer"
        elif use_interro and not use_answer:
            setting="-interro-nonanswer-repanswer"
        elif not use_interro and not use_answer:
            setting="-nonanswer-repanswer"
    else:
        if use_interro and not use_answer:
            setting="-interro"
        elif not use_interro and use_answer:
            setting="-answer"
        elif use_interro and use_answer:
            setting="-interro-answer"
        elif not use_interro and not use_answer:
            setting="-normal"
        elif use_interro==True and use_pre_interro==True:
            setting="-preinterro"


    if train==True:
        random_list=list(range(len(questions)))
        #random.shuffle(random_list)
        with open("data/squad-src-train{}.txt".format(setting),"w")as f:
            for i in random_list:
                f.write(sentences[i]+"\n")
        with open("data/squad-tgt-train{}.txt".format(setting),"w")as f:
            for i in random_list:
                f.write(questions[i]+"\n")
        with open("data/squad-ans-train{}.txt".format(setting),"w")as f:
            for i in random_list:
                f.write(answers[i]+"\n")
        with open("data/squad-interro-train{}.txt".format(setting),"w")as f:
            for i in random_list:
                f.write(interros[i]+"\n")
        with open("data/squad-noninterro-train{}.txt".format(setting),"w")as f:
            for i in random_list:
                f.write(non_interros[i]+"\n")

    if train==False:
        random_list=list(range(len(questions)))
        #random.shuffle(random_list)
        val_num=int(len(random_list)*0.5)
        with open("data/squad-src-val{}.txt".format(setting),"w")as f:
            for i in random_list[0:val_num]:
                f.write(sentences[i]+"\n")
        with open("data/squad-tgt-val{}.txt".format(setting),"w")as f:
            for i in random_list[0:val_num]:
                f.write(questions[i]+"\n")
        with open("data/squad-ans-val{}.txt".format(setting),"w")as f:
            for i in random_list[0:val_num]:
                f.write(answers[i]+"\n")
        with open("data/squad-interro-val{}.txt".format(setting),"w")as f:
            for i in random_list[0:val_num]:
                f.write(interros[i]+"\n")
        with open("data/squad-noninterro-val{}.txt".format(setting),"w")as f:
            for i in random_list[0:val_num]:
                f.write(non_interros[i]+"\n")

        with open("data/squad-src-test{}.txt".format(setting),"w")as f:
            for i in random_list[val_num:]:
                f.write(sentences[i]+"\n")
        with open("data/squad-tgt-test{}.txt".format(setting),"w")as f:
            for i in random_list[val_num:]:
                f.write(questions[i]+"\n")
        with open("data/squad-ans-test{}.txt".format(setting),"w")as f:
            for i in random_list[val_num:]:
                f.write(answers[i]+"\n")
        with open("data/squad-interro-test{}.txt".format(setting),"w")as f:
            for i in random_list[val_num:]:
                f.write(interros[i]+"\n")
        with open("data/squad-noninterro-test{}.txt".format(setting),"w")as f:
            for i in random_list[val_num:]:
                f.write(non_interros[i]+"\n")





if __name__ == "__main__":
    random.seed(0)

    data_process(input_path="data/squad-dev-v1.1.json",
                interro_path="data/squad-data-dev.json",
                train=False
                )

    data_process(input_path="data/squad-train-v1.1.json",
                interro_path="data/squad-data-train.json",
                train=True
                )
