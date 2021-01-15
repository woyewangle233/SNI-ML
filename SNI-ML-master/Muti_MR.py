
import copy
from config.config import args
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import heapq
import os
def getfile_story(dir,Filelist):
    if os.path.isfile(dir):
        if True:
            Filelist.append(dir)
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newDir = os.path.join(dir, s)
            getfile_story(newDir,Filelist)
    return Filelist
def cosine_similarity(x,y):
    num = x.dot(y.T)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom==0:
        return 0
    cos=num / denom
    return cos

def equal_(a, b):
    for i in range(len(a)):
        if ( (abs(a[i]-b[i])) < 0.001):
            pass
        else:
            return False
    return True
def update_s(i,lam, salience_list, graph, ):       # si = λ· ∑j̸=i  sj·p(j, i) + (1−λ)·1/|V|
    m=0
    for k in range(len(salience_list)):             # j=k
        if k!=i:
            m=m+salience_list[k]*graph[k][i]

    m=lam*m+(1-lam)/len(salience_list)
    return m
def get_name(file):
    a=re.sub('.*\\\\','',file)
    a=re.sub('\.txt','',a)
    a=re.sub('\.story','',a)
    a = re.sub('\.sum', '', a)
    return a

def tf_idf_encoder(file_name):
    f = open(file_name, 'r', encoding='utf-8')
    sentences = f.readlines()

    vectorizer = CountVectorizer(max_features=10000)
    tf_idf_transformer = TfidfTransformer()
    tf_idf = tf_idf_transformer.fit_transform(vectorizer.fit_transform(sentences))
    sentences_array = tf_idf.toarray()

    return [],sentences_array

def MMR(array,already_write):
    if already_write==[]:
        return True
    else:
        for a in already_write:
            if cosine_similarity(array,a)>0.2:
                return False
        return True

def noramlization(data):
    data = [float(i) for i in data]
    arr = np.asarray(data)
    a = []
    div = (np.max(arr) - np.min(arr))
    if div == 0:
        return [0 for i in range(len(data))]
    for x in arr:
        x = float(x - np.min(arr)) / div
        a.append(x)
    return a


def social_score(file_name):
    name=get_name(file_name)
    path=args.sn

    f = open(path + name + '.num', 'r', encoding='utf-8')
    scores_like=[]
    scores_retweet=[]
    scores_comment=[]
    for line in f.readlines():
        line=re.sub('\n','',line)
        score_like=line.split('\t')[2]
        score_retweet = line.split('\t')[1]
        score_comment = line.split('\t')[0]
        scores_like.append(score_like)
        scores_retweet.append(score_retweet)
        scores_comment.append(score_comment)
    f.close()

    scores_like = noramlization(scores_like)
    scores_retweet = noramlization(scores_retweet)
    scores_comment =noramlization(scores_comment)
    s=[]
    for i in range(len(scores_like)):
        score=0.6*scores_comment[i]+0.2*scores_retweet[i]+0.2*scores_like[i]
        s.append(score)

    return s
def splus(s,file_name,lamada):
    social_s = []
    s = noramlization(s)
    social = social_score(file_name)
    # lamada=0
    for i in range(len(s)):
        score = s[i] * lamada + social[i] *(1-lamada )
        social_s.append(score)

    return social_s

def MR_S(array,docs):
    W_a=np.zeros([len(array),len(array)])
    for i in range(len(array)):
        for j in range(i+1,len(array)):
            if doc_same(docs,i,j):
                W_a[i][j]=cosine_similarity(array[i],array[j])
    for i in range(len(array)):
        for j in range(i,len(array)):
            W_a[j,i]=W_a[i][j]
    W_b = np.zeros([len(array), len(array)])
    for i in range(len(array)):
        for j in range(i + 1, len(array)):
            if not doc_same(docs, i, j):
                W_b[i][j] = cosine_similarity(array[i], array[j])
    for i in range(len(array)):
        for j in range(i, len(array)):
            W_b[j, i] = W_b[i][j]
    data=0.3*W_a+1*W_b               # calculate W,  lamada_1,lamada_2 ∈ [0,1]

    l = np.zeros(len(array))
    for k in range(len(l)):  # 求 S
        l[k] = np.sum(data[k][:])
        if l[k] != 0:
            l[k] = 1 / np.sqrt(l[k])
    D_1 = np.diag(l)
    S = np.dot(np.dot(D_1, data), D_1)
    return S


def get_filed_word():
    files=[]
    filed_word = args.words
    getfile_story(filed_word,files)
    theme_words =[]
    titles=[]
    for file_name in files:
        title=re.sub('.*\\\\','',file_name)
        titles.append(title)
        f=open(file_name,'r',encoding='utf-8')
        a=f.readlines()
        a=[re.sub('\n','',w) for w in a]
        theme_words.append(a)
    return theme_words,titles
def Doc(file_name):             #   Get multiple documents
    _,titles=get_filed_word()
    f=open(file_name,'r',encoding='utf-8')
    lines= f.readlines()
    f.close()

    all_line = [[] for i in range(len(titles))]
    for i in range(len(lines)):
        for k in range(len(titles)):
            if titles[k] in lines[i]:
                all_line[k].append(i)

    point=0
    while point!=1:
        try:
            all_line.remove([])
        except:
            point=1
            pass
    return all_line
def doc_same(docs,i,j):  #docs [[doc0],[doc1],[doc2]]    doc0=0,1,2 #Determine whether the sentence i, j is a doc text
    for doc in docs:
        if i in doc:
            if j in doc:
                return True
            else:
                return False
def S_a(array,docs):
    W_a=np.zeros([len(array),len(array)])
    for i in range(len(array)):
        for j in range(i+1,len(array)):
            if doc_same(docs,i,j):
                W_a[i][j]=cosine_similarity(array[i],array[j])
    for i in range(len(array)):
        for j in range(i,len(array)):
            W_a[j,i]=W_a[i][j]

    l = np.zeros(len(array))
    for k in range(len(l)):  #  S
        l[k] = np.sum(W_a[k][:])
        if l[k] != 0:
            l[k] = 1 / np.sqrt(l[k])
    D_1 = np.diag(l)
    S_a = np.dot(np.dot(D_1, W_a), D_1)
    return S_a
def S_b(array,docs):
    W_b=np.zeros([len(array),len(array)])
    for i in range(len(array)):
        for j in range(i+1,len(array)):
            if not doc_same(docs,i,j):
                W_b[i][j]=cosine_similarity(array[i],array[j])
    for i in range(len(array)):
        for j in range(i,len(array)):
            W_b[j,i]=W_b[i][j]

    l = np.zeros(len(array))
    for k in range(len(l)):  # 求 S
        l[k] = np.sum(W_b[k][:])
        if l[k] != 0:
            l[k] = 1 / np.sqrt(l[k])
    D_1 = np.diag(l)
    S_b = np.dot(np.dot(D_1, W_b), D_1)

    return S_b

def Div_penalty(S,f):         # MMR algorithm returns the sorted sentence id, Hyperparameter omiga
    id = []
    omiga = 8
    Rank_score = copy.deepcopy(f)
    rank_id = sorted(range(len(f)), key=lambda k: f[k], reverse=True)
    id.append(rank_id[0])
    for m in range(1, len(f)):
        for i in id:
            for j in rank_id[m:]:               # j = rank_id[m]
                Rank_score[j] = Rank_score[j] - omiga * S[j][i] * f[i]
        rank_id = sorted(range(len(Rank_score)), key=lambda k: Rank_score[k], reverse=True)
        id.append(rank_id[m])
    return  Rank_score    #return id
def Manifold_Rank(path):                        #Summary method based on the internal relationship between the topic and the sentence
    s, array = tf_idf_encoder(path)
    docs=Doc(path)
    S = MR_S(array,docs)
    # print(S)
    update = [np.zeros(len(S)), np.ones(len(S))]
    alpha = 0.7                        #Hyperparameters, the smaller the rank result, the closer to the topic
    y = np.array([1  for i in range(len(S))]).T
    f = y
    while (equal_(update[-1], update[-2]) is False):
        f = alpha * np.dot(S, f) + (1 - alpha) * y
        update.append(copy.deepcopy(f))                   # f is the score of each sentence
    socre=Div_penalty(S,f)
    return socre

def Muti_MR_li(file_name,):                 #Linear Fusion (Multi-Modality Learning) Returns the significance score (numpy format)
    s, array = tf_idf_encoder(file_name)
    l=len(array)
    docs=Doc(file_name)
    Sa=S_a(array,docs)
    Sb=S_b(array,docs)
    mu = args.mu  # Hyperparameter，μ，η   lamada
    eta = args.eta
    update = [np.zeros(l), np.ones(l)]
    y = np.array([1  for i in range(l)]).T
    f = y
    while (equal_(update[-1], update[-2]) is False):
        f = mu * np.dot(Sa, f) +eta*np.dot(Sb, f)+ (1 - mu-eta) * y
        update.append(copy.deepcopy(f))

    socre=update[-1]
    # socre=Div_penalty(Sa+Sb,f)

    return socre

def Muti_MR_com(file_name,):                        #Score Combination (Multi-Modality Learning) Returns the significance score (numpy format)
    s, array = tf_idf_encoder(file_name)
    l=len(array)
    docs=Doc(file_name)
    Sa=S_a(array,docs)
    Sb=S_b(array,docs)
    mu=args.mu          #Hyperparameter，μ，η   lamada
    eta=args.eta
    lam=args.lam
    update = [np.zeros(l), np.ones(l)]
    y = np.array([1  for i in range(l)]).T
    f1 = y
    f2=copy.deepcopy(y)
    while (equal_(update[-1], update[-2]) is False):
        f1 = mu * np.dot(Sa, f1) + (1 - mu) * y
        update.append(copy.deepcopy(f1))
    update=[np.zeros(l), np.ones(l)]
    while (equal_(update[-1], update[-2]) is False):
        f2 = eta * np.dot(Sb, f2) + (1 - eta) * y
        update.append(copy.deepcopy(f2))
    f=lam*f1+(1-lam)*f2
    socre=f
    return socre,array


def write2sum(file_name,write_path,F,lamada):      # file_name, the story file to be summarized write_path write path F name of the function called
    f=open(file_name,'r',encoding='utf-8')
    lines=f.readlines()
    max=copy.deepcopy(int(len(lines)*0.2))
    f.close()
    mm = re.sub('\.story', '', file_name)
    mm = re.sub('\.txt', '', mm)
    name = re.sub('.*\\\\', '', mm)

    if not os.path.exists(write_path+name+'.txt'):
        s,array = F(file_name)
        s=list(s)               # No social network information

        s = splus(s, file_name,lamada)
        a = map(s.index, heapq.nlargest(len(s), s))
        f2 = open(write_path + name + '.txt', 'w', encoding='utf-8')
        num = 0
        for i in a:
            f2.write(lines[i])
            num += 1
            if num == max:
                break
        f2.close()
        return

