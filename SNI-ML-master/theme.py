import math
import numpy as np
import gensim
import re
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import heapq
from config.config import args
import os

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
def get_name(file):
    a=re.sub('.*\\\\','',file)
    a=re.sub('\.txt','',a)
    a=re.sub('\.story','',a)
    a = re.sub('\.sum', '', a)
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

def getfile(dir,Filelist):
    if os.path.isfile(dir):
        Filelist.append(dir)
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newDir = os.path.join(dir, s)
            getfile(newDir,Filelist)
    return Filelist
def get_filed_word():
    files=[]
    filed_word = './filed_words'
    getfile(filed_word,files)
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
theme_words,titles=get_filed_word()

word_model = gensim.models.KeyedVectors.load_word2vec_format(
            args.vector,
            limit=1000000, binary=True)
def cosine_similarity(x,y):
    num = x.dot(y.T)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    cos=num / denom
    return cos
def __func_preprocess(string):   #  句子string的前处理
    # 去除字符串中的标点符号
    string = re.sub(' .. ',' ',string)
    string = re.sub(pattern="[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。🤔😎👌？、~@#￥%……&*（）)]+", repl=" ", string=string)
    tokens_word_only = re.findall(r'[a-z]+', string)
    tokens_illegal_free = [word for word in tokens_word_only if bool(wordnet.synsets(word))]
    tokens_char_free = [word for word in tokens_illegal_free if len(word) > 1]
    tokens_stop_free = [word for word in tokens_char_free if word not in stopwords.words('english')]
    tokens_c_pos = nltk.pos_tag(tokens_stop_free,)
    word_list=[]
    for word,tag in tokens_c_pos:
        if  'NN' or 'JJ' or 'VB' in tag:
            word_list.append(word)
    return word_list                # 返回以列表形式表示的句子（保留：只保留名词动词形容词）
def getfile_story(dir,Filelist):
    if os.path.isfile(dir):
        Filelist.append(dir)
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newDir = os.path.join(dir, s)
            getfile_story(newDir,Filelist)
    return Filelist
def sentence_2_theme_similarity(line,theme_index):       # 计算句子到 主题的相似度
    sentence=set(__func_preprocess(line))           #set 减少重复
    length=10
    if len(sentence)==0:
        return 0
    similar_arr = []

    for word in sentence:
        word2theme_similar = []
        for theme_word in theme_words[theme_index]:
            try:
                a = cosine_similarity(word_model[theme_word], word_model[word])
            except:a=0
            word2theme_similar.append(a)
        word2theme=max(word2theme_similar)
        similar_arr.append(word2theme)
    arr=np.mean(similar_arr)
    return arr
def MMR(sentence,summary):            #最大边际相关——保证多样性
    sent=__func_preprocess(sentence)
    for s in summary:
        s=(__func_preprocess(s))
        if len(list(set(s) & set(sent)))>3:
            return False
    return True
def theme_score(file):
    if True:
        filename=file
        with open(filename, 'r', encoding='utf-8') as f:
            all_line = [[] for i in range(len(titles))]
            for line in f.readlines():
                for k in range(len(titles)):
                    if titles[k] in line:
                        all_line[k].append(line)
                        break
        name = re.sub('.*\\\\', '', filename)
        name = re.sub('\.story', '', name)
        theme_index = -1

        str_score = []
        for story in all_line:
            theme_index += 1
            score = []
            sum = []
            word_lenth = 0
            for line in story:
                word_lenth += len(line.split())

            for line in story:
                a = sentence_2_theme_similarity(line, theme_index)  # 相似度计算
                score.append(a)
                sum.append(line)
            max = int(len(sum) * 0.2)
            most_similar_index = map(score.index, heapq.nlargest(len(score), score))
            summary = []
            score_list = []

            t = 0
            for i in most_similar_index:
                if MMR(sum[i], summary):
                    t += 1
                    summary.append(sum[i])
                    score_list.append(score[i])
                if t == max:
                    break
            if score_list != []:
                str_score.append(np.average(score_list))
            else:
                str_score.append(0)

        return str_score

def Generate_theme_score(files,write_path):                # 通过和主题的 相似度计算silience score    #应用MMR后形成摘要
    countnum =0
    for filename in files:
        if countnum % 10 == 0:
            print(countnum,'\t',filename)
            print()
        countnum += 1
        if countnum<0:           #已经跑完的不用再跑
            continue
        with open(filename, 'r', encoding='utf-8') as f:
            all_line=[[] for i in range(len(titles))]
            for line in f.readlines():
                for k in range(len(titles)):
                    if titles[k] in line:
                        all_line[k].append(line)
                        break
        path =write_path                          # 候选摘要保存路径
        name = re.sub('.*\\\\', '', filename)
        name = re.sub('\.story', '', name)
        fname =path + name + '.npy'
        theme_index=-1
        social_s=social_score(filename)
        # print(social_s)
        alpha=0.7          #
        beta=0.2
        count=0

        str_score=[]
        for story in all_line:
            theme_index+=1
            score = []
            sum = []
            word_lenth=0
            for line in story:
                word_lenth+=len(line.split())

            i=0

            for line in story:
                a = sentence_2_theme_similarity(line,theme_index)   #    相似度计算
                i=i+1
                score.append(a)
                sum.append(line)
            max=int(len(sum)*0.2)
            most_similar_index = map(score.index, heapq.nlargest(len(score), score))
            summary=[]
            score_list=[]

            t = 0
            for i in most_similar_index:
                if MMR(sum[i], summary):
                    t += 1
                    summary.append(sum[i])
                    score_list.append(score[i])
                if t == max:
                    break
            if score_list!=[]:
                str_score.append(np.average(score_list))
            else:
                str_score.append(0)

        np.save(fname,np.array(str_score))


