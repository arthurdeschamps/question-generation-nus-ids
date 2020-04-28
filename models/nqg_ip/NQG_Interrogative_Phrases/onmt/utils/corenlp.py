from pycorenlp import StanfordCoreNLP
from tqdm import tqdm

#corenlpを用いて疑問詞とその周辺を取り出す
#ex. How many icecream did you eat? -? How many icecream
#疑問詞を特定できないもの(interro_listの中のタグがあわられないもの)は除去


class CoreNLP():
    def __init__(self):
        self.nlp=StanfordCoreNLP('http://localhost:9000')
        self.interro_list=["WDT","WP"," WP$","WRB","VB","VBD","VBG","VBN","VBP","VBZ"]
        self.count=-1

    #textの疑問詞句を抽出する
    #疑問詞句のリストと、それ以外の単語のリストを返す
    def forward(self,text):
        q=self.nlp.annotate(text, properties={'annotators': 'tokenize,ssplit,parse','outputFormat': 'json'})

        tokens=q["sentences"][0]["tokens"]              #文の中の単語
        deps=q["sentences"][0]["basicDependencies"]     #依存関係（使わない）
        parse_text=q["sentences"][0]["parse"]           #句構造のパース結果

        token_list=[{"index":token["index"],"text":token["originalText"],"pos":token["pos"]} for token in tokens]
        parse_text=parse_text.replace("(","( ").replace(")"," )").split()

        WP_list=[]      #疑問詞句に含まれる単語
        NotWP_list=[]   #疑問詞句に含まれない単語
        WP_flag=False   #疑問詞句を発見済みならTrue

        depth=0
        for i in range(len(parse_text)-1):
            #depthが0の場合かつ、（疑問詞の句構造に未突入、または、すでに疑問詞が見つかっている）
            if depth==0 and ("WH" not in parse_text[i] or WP_flag==True):
                if parse_text[i]!=")" and parse_text[i+1]==")":
                    NotWP_list.append(parse_text[i])
                continue
            #疑問詞の句構造の内部にいる時
            else:
                WP_flag=True
                depth=max(depth,1)
                if parse_text[i]=="(":
                    depth+=1
                elif parse_text[i]==")":
                    depth-=1
                if parse_text[i]!=")" and parse_text[i+1]==")":
                    WP_list.append(parse_text[i])

        return WP_list,NotWP_list

    def verb_check(self,text):
        q=self.nlp.annotate(text, properties={'annotators': 'tokenize,ssplit,pos','outputFormat': 'json'})
        pos_list=[]
        for sentence in q["sentences"]:
            for token in sentence["tokens"]:
                pos_list.append(token["pos"])

        for pos in pos_list:
            if "VB" in pos:
                return True
        return False

    def forward_verbcheck(self,text):#input:(batch,seq_len)
        self.count+=1
        q=self.nlp.annotate(text, properties={'annotators': 'tokenize,ssplit,parse','outputFormat': 'json'})

        tokens=q["sentences"][0]["tokens"]
        deps=q["sentences"][0]["basicDependencies"]

        token_list=[]
        token_list.append({"index":0,"text":"ROOT"})
        interro_id=-1
        for token in tokens:
            token_list.append({"index":token["index"],"text":token["originalText"],"pos":token["pos"]})
        for token in tokens:
            if interro_id==-1 and token["pos"] in self.interro_list[0:4]:#疑問詞のチェック
                interro_id=token["index"]
        for token in tokens:
            if interro_id==-1 and token["pos"] in self.interro_list[4:]:#疑問詞のチェック
                interro_id=token["index"]

        #疑問詞がなかった時のエラー処理
        if interro_id==-1:
            #print(self.count)
            return "none_tag",text,False

        #vb
        pos_list=[]
        for sentence in q["sentences"]:
            for token in sentence["tokens"]:
                pos_list.append(token["pos"])

        vb_check=False

        for pos in pos_list:
            if "VB" in pos:
                 vb_check=True

        G = nx.DiGraph()
        G.add_nodes_from(range(len(token_list)))
        for dep in deps:
            G.add_path([dep["dependent"],dep["governor"]])
        if nx.has_path(G,interro_id,0)==False:
            print("error")
        s_path=nx.shortest_path(G,interro_id,0)


        if len(s_path)==2:#疑問詞だけ
            node_list=[s_path[0]]
        else:#疑問詞周り
            node_list=[node for node in G.nodes() if nx.has_path(G,node,s_path[-3])]
        neg_node_list=[node for node in G.nodes() if node not in node_list and node!=0]
        question=" ".join([token_list[node]["text"] for node in node_list])
        neg_question=" ".join([token_list[node]["text"] for node in neg_node_list])
        return question,neg_question,vb_check

    #charactor単位でのそれぞれの文のスタートとエンドの位置を返す
    #[(0,10),(11,35),(35,60)]など
    def sentence_tokenize(self,text):
        #テキストの一番最初に空白、改行がある場合はインデックスがずれるのでそれの処理
        space_count=0
        while True:
            if text[space_count]==" " or text[space_count]=="\n":
                space_count+=1
            else:
                break
        #print(space_count)
        q=self.nlp.annotate(text, properties={'annotators': 'tokenize,ssplit','outputFormat': 'json'})
        sentences=[]
        for sentence in q["sentences"]:
            tokens=sentence["tokens"]#文の中の単語
            start_id=tokens[0]["characterOffsetBegin"]+space_count
            end_id=tokens[-1]["characterOffsetEnd"]+space_count
            sentences.append((start_id,end_id))

        return sentences
