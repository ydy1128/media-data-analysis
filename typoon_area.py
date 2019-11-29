#%%
import re
import numpy as np
import sys
import networkx as nx
import copy
from networkx.algorithms import community
import matplotlib.animation as animation
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib import rc

# 설정
font_path="./font/malgun.ttf"
font_name = fm.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)

df = pd.read_csv("./data/태풍_location_info_행정구.csv")
'''
hashtag_keywords = ["경로","피해","태풍","링링","타파","바람","비바람","속도","호우","기상청","저기압","풍속" \
            "폭풍","강풍","풍량","주의보","수해","침수","재해"]
contains_df = df[df.comment_hashtag.apply(lambda x : any([(w in x) for w in hashtag_keywords]))]
'''

# 자연어처리
from eunjeon import Mecab
tagger = Mecab()
#fullcomments = df['행정구_depth2'][0:10]
fullcomments = df['행정구_depth3']
insta_nouns = []
for li in fullcomments:
    try:
        for noun in tagger.nouns(li):
            if noun not in comments_stop_words:
                insta_nouns.append(noun)
    except: 
        pass

# 전체컨텐츠
comments = fullcomments
num_top_nouns = 100
insta_nouns_counter = Counter(insta_nouns)
insta_top_nouns = dict(insta_nouns_counter.most_common(num_top_nouns))

insta_sentences_nouns = []
for sentence in insta_sentences:
    sentence_nouns = tagger.nouns(sentence)
    insta_sentences_nouns.append(sentence_nouns)
#insta_sentences_nouns[0:10]

# 키부여
insta_word2id = {w: i for i, w in enumerate(insta_top_nouns.keys())}
print(dict(insta_word2id))

insta_id2word = {i: w for i, w in enumerate(insta_top_nouns.keys())}
#insta_id2word

# 행렬생성
insta_adjacent_matrix = np.zeros((num_top_nouns, num_top_nouns), int)
for sentence in insta_sentences_nouns:
    for wi, i in insta_word2id.items():
        if wi in sentence:
            for wj, j in insta_word2id.items():
                if i != j and wj in sentence:
                    insta_adjacent_matrix[i][j] += 1
#insta_adjacent_matrix

insta_network = nx.from_numpy_matrix(insta_adjacent_matrix)
#list(insta_network.adjacency())

# Graph행렬
testG = nx.Graph() # 단어행렬

# Graph에 node와 edge를 삽입
# add_weighted_edges_from는 (from_node, to_node, weight) 3-tuple 
# Graph에 edge로 추가하면 (from_node, to_node, {'weight':weight}) 형태로 바뀐다.

data = list(insta_network.adjacency())
for u in data:
    testG.add_nodes_from(
        [ ("{}".format(insta_id2word[u[0]]), {'weight':len(u[1])})]
    )
 

for u in data:
    for v in u[1:]:
        for c1, c2 in v.items():
            testG.add_weighted_edges_from(
                [ ("{}".format(insta_id2word[u[0]]), "{}".format(insta_id2word[c1]), c2['weight'])]
            )

print("--- Step 1 End---")
from community import community_louvain
print("--- community_louvain_scale ---")
def draw_whole_graph(inputG, outPicName):
    #plt.close('all')
    #plt.margins(x=0.05, y=0.05) # text 가 잘리는 경우가 있어서, margins을 넣음
    #pos = nx.spring_layout(inputG)
    #pos = nx.random_layout(inputG)
    #pos = nx.circular_layout(inputG)
    pos = nx.kamada_kawai_layout(inputG)

    """
    한번씩 input_lst가 비어있을때가 있는데 왜 그런지 확인 필요.
    """    
    def return_log_scaled_lst(input_lst):
        r_lst = map(np.log, input_lst)
        try:
            max_v = max(map(np.log, input_lst))
            min_v = min(map(np.log, input_lst))
            return map(lambda v: v/max_v, r_lst)
        except: 
            print(input_lst)
    node_weight_lst = return_log_scaled_lst([n[1]['weight'] for n in inputG.nodes(data=True)])
    edge_weight_lst = return_log_scaled_lst([e[2]['weight'] for e in inputG.edges(data=True)])
    print(edge_weight_lst)
    # label의 경우는 특정 node만 그릴 수 없음. 그리면 모두 그려야함.   
    # 루뱅파티션
    partition = community_louvain.best_partition(testG)  # compute communities
    plt.figure(figsize=(20, 20))
    plt.axis('off')
    print(edge_weight_lst)
    nx.draw_networkx_nodes(inputG, pos, node_size=list(map(lambda x: x*3000, node_weight_lst)), cmap=plt.cm.RdYlBu, node_color=list(partition.values()))
    nx.draw_networkx_edges(inputG, pos, alpha=1, edge_color='black', width = list(map(lambda x: 0.3*2**(x+1), edge_weight_lst)))
    nx.draw_networkx_labels(inputG, pos, font_size=8, font_color="black", font_family=font_name)
    plt.savefig('./data/'+outPicName)    
    plt.show(testG)


draw_whole_graph(testG,'community_louvain_scale_area')
print("--- Step 2 End---")

# %%
