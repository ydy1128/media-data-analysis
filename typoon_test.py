#%%
import re
import numpy as np
import networkx as nx
import sys

#%%
import pandas as pd

df = pd.read_csv("./data/태풍링링_comment_info.csv")
df2 = pd.read_csv("./data/태풍링링_post_info.csv")


hashtag_keywords = ["경로","피해","태풍","링링","타파","바람","비바람","속도","호우","기상청","저기압","풍속" \
            "폭풍","강풍","풍량","주의보","수해","침수","재해"]
contains_df = df[df.comment_hashtag.apply(lambda x : any([(w in x) for w in hashtag_keywords]))]
contains_df2 = df2[df2.CaptionHashtag.apply(lambda x : any([(w in x) for w in hashtag_keywords]))]
coments_keywords = [ \
"담장","담벽","비상구","환풍구","비상","출입구","소화시설", "공사","지하철","호텔", \
"학교","교통","안전","시설","신호등","중앙분리대","표지판","도로","중앙선","과속", \
"노인","마을","회관","병원","놀이","계단","게임","체육관", "농업","저수지", \
"논지","경계","가드레일","휀스","울타리","팬스","펜스","가림판","가로수","화단", \
"방풍","보도","블럭","난간","가로등","소화전","보안","블록","소방","방지", \
"낙석","전기","전주","전봇대","전신주","전선","통신","한전","박스","변압기", \
"분전함","배전","선로","계량","하수","맨홀","배수","그레이팅","하수","집수정", \
"처리장","배수관","우수","집수정","하수도관","오수관","받이","빗물","우수","배관", \
"관로","보행자","횡단보도","육교","교량","인도","자동차","다리","차도","교량", \
"고속","교량","외곽","사거리","로터리","자전거","정류장","승강장","터미널","여객", \
"버스","시외","주거","아파트","주택","빌라","단지","센터","주차장","공영",\
"휴식","공원","산책","등산","유원지","팔각정","쉼터","공장","건물","빌딩"]

#hashtags = contains_df.comment_hashtag.apply(lambda x : x[1:-1].split(", ")).apply(lambda x : [w[2:-1] for w in x])

#%%
from eunjeon import Mecab
tagger = Mecab()

#fullcomments = contains_df.comment + contains_df.comment_hashtag
#fullcomments = contains_df2.caption
fullcomments = pd.concat([contains_df.comment,contains_df2.caption])
#fullcomments = fullcomments[0:100]
#print(fullcomments)
fillter = "|".join(coments_keywords)
#print(fullcomments.str.contains(fillter))
comments = fullcomments[fullcomments.str.contains(fillter)]
#comments = comments[0:100]
#print(comments)


#%%
comments_stop_words = "은 이 것 등 더 를 좀 즉 인 옹 때 만 원 이때 개 일 기 ㄷ 호 줄 시 거 듯 년 월 번 이번 견 게"
comments_stop_words = comments_stop_words.split(' ')
comments_stop_words[0:10]

insta_nouns = []
for li in comments:
    for noun in tagger.nouns(li):
        if noun not in comments_stop_words:
            insta_nouns.append(noun)
            
insta_nouns[0:10]
#%%

insta_nouns = coments_keywords

from collections import Counter
num_top_nouns = 115

insta_nouns_counter = Counter(insta_nouns)
insta_top_nouns = dict(insta_nouns_counter.most_common(num_top_nouns))


#%%
insta_sentences = []
for post in comments:
    insta_sentences.extend(re.split('; |\.|\?|\!', post))
insta_sentences[0:10]


#%%
insta_sentences_nouns = []
for sentence in insta_sentences:
    sentence_nouns = tagger.nouns(sentence)
    insta_sentences_nouns.append(sentence_nouns)
insta_sentences_nouns[0:10]

#%%
insta_word2id = {w: i for i, w in enumerate(insta_top_nouns.keys())}
insta_word2id


#%%
insta_id2word = {i: w for i, w in enumerate(insta_top_nouns.keys())}
insta_id2word


#%%
import numpy as np
insta_adjacent_matrix = np.zeros((num_top_nouns, num_top_nouns), int)
for sentence in insta_sentences_nouns:
    for wi, i in insta_word2id.items():
        if wi in sentence:
            for wj, j in insta_word2id.items():
                if i != j and wj in sentence:
                    insta_adjacent_matrix[i][j] += 1
insta_adjacent_matrix


#%%
insta_network = nx.from_numpy_matrix(insta_adjacent_matrix)
list(insta_network.adjacency())

#%%
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib import rc

font_path="./font/malgun.ttf"
font_name = fm.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)

fig = plt.figure()
fig.set_size_inches(10, 10)
ax = fig.add_subplot(1, 1, 1)
ax.axis("off")
option = {
    'node_color' : 'lightblue',
    'node_size' : 1000,
    'size' : 100
}
#nx.draw(insta_network, labels=insta_id2word, font_family=font_name, ax=ax, **option)
nx.draw_random(insta_network, labels=insta_id2word, font_family=font_name, **option)

#%%
'''
print(list(insta_network.adjacency())[:10])
k = list(insta_network.adjacency())[:10]

for u in k:
    #print(u)
    for a in u[1:]:
        #print(u[0],a)
        for key, val in a.items():
            print(u[0],key,val['weight'])
'''
# %%
import networkx as nx
import numpy as np 
from community import community_louvain
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

def drop_low_weighted_edge(inputG, above_weight=3):
    rG = nx.Graph()
    rG.add_nodes_from(inputG.nodes(data=True))
    edges = filter(lambda e: True if e[2]['weight']>=above_weight else False, inputG.edges(data=True))
    rG.add_edges_from(edges)
    """
    neighbor가 없는 isolated node를 모두 지운다. 
    """
    for n in inputG.nodes():
        if len(list(nx.all_neighbors(rG, n)))==0:
            rG.remove_node(n)
        #print(n, list(nx.all_neighbors(rG, n)))
    return rG

font_path="./font/malgun.ttf"
font_name = fm.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)


graph_size = 115
testG = nx.Graph()

#data = list(insta_network.adjacency())[:5]
data = list(insta_network.adjacency())

for u in data:
    #print(u)
    #print(insta_id2word[u[0]],len(u[1]))
    #print(u[0],len(u[1]))    
    testG.add_nodes_from(
        [ ("{}".format(insta_id2word[u[0]]), {'weight':len(u[1])})]
    )  
    

for u in data:
    #print(u)
    for v in u[1:]:
        #print(insta_id2word[u[0]],v)
        #print(u[0],a)
        for c1, c2 in v.items():
            #print(u[0],c1,c2['weight'])
            testG.add_weighted_edges_from(
                [ ("{}".format(insta_id2word[u[0]]), "{}".format(insta_id2word[c1]), c2['weight'])]
            )

        
            


for e in testG.edges(data=True):
    print(e)

for h in testG.nodes(data=True):
    print(h)


#%%

#add_weighted_edges_from는 (from_node, to_node, weight) 3-tuple 
#Graph에 edge로 추가하면 (from_node, to_node, {'weight':weight}) 형태로 바뀐다.
'''
testG.add_weighted_edges_from(
    [ ("{}".format(i), "{}".format(j), np.random.randint(1.0, 10.0)) # (from_node, to_node, weight) 
     for i in range(0, graph_size) for j in range(0, graph_size)]
)
'''

partition = community_louvain.best_partition(testG)  # compute communities
print(partition.values())
#pos = nx.spring_layout(testG)  # compute graph layout
pos = nx.random_layout(testG)  # compute graph layout
plt.figure(figsize=(7, 7))



plt.axis('off')
nx.draw_networkx_nodes(testG, pos, node_size=1000, cmap=plt.cm.RdYlBu, node_color=list(partition.values()))
nx.draw_networkx_edges(testG, pos, alpha=1, width=1, edge_color='white')
'''
nx.draw_networkx_edges(testG, pos, 
                        width = list(map(lambda x: 5**(x+1), edge_weight_lst)), 
                        edge_color='white', alpha=1
                        )
'''
nx.draw_networkx_labels(testG, pos, font_size=8, font_color="black", font_family=font_name)
plt.show(testG)

#testG = drop_low_weighted_edge(testG, 1)
pos = nx.random_layout(testG)  # compute graph layout
plt.axis('off')
nx.draw_networkx_nodes(testG, pos, node_size=1000, cmap=plt.cm.RdYlBu, node_color=list(partition.values()))
nx.draw_networkx_edges(testG, pos, alpha=1, edge_color='white')
nx.draw_networkx_labels(testG, pos, font_size=8, font_color="black", font_family=font_name)
plt.show(testG)



# %%
def draw_whole_graph(inputG, outPicName):
    plt.close('all')
    plt.margins(x=0.05, y=0.05) # text 가 잘리는 경우가 있어서, margins을 넣음
    pos = nx.spring_layout(inputG)

    
    #print(inputG.nodes(data=True))

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
    # label의 경우는 특정 node만 그릴 수 없음. 그리면 모두 그려야함.   
    partition = community_louvain.best_partition(testG)  # compute communities
    plt.figure(figsize=(20, 20))
    plt.axis('off')
    nx.draw_networkx_nodes(inputG, pos, node_size=list(map(lambda x: x*3000, node_weight_lst)), cmap=plt.cm.RdYlBu, node_color=list(partition.values()))
    nx.draw_networkx_edges(inputG, pos, alpha=1, edge_color='white', width = list(map(lambda x: 2**(x+1), edge_weight_lst)))
    nx.draw_networkx_labels(inputG, pos, font_size=10, font_color="white", font_family=font_name)
    plt.savefig('./data/'+outPicName)
    plt.show(testG)


draw_whole_graph(testG,'result')


# %%
