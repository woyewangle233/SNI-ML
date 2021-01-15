

import re
from sumeval.metrics.rouge import RougeCalculator
import os
import numpy as np
from config.config import args
from theme import *
def getfile_story(dir,Filelist):
    if os.path.isfile(dir):
        if True :
            Filelist.append(dir)
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newDir = os.path.join(dir, s)
            getfile_story(newDir,Filelist)
    return Filelist
def get_relevance(file):
    a=theme_score(file[0])
    b = np.load(file[1],allow_pickle=True)
    x=0
    count=0
    for i in range(9):
        if float(b[i])!=0:
            count+= 1
            x = x + abs(float(a[i]) - float(b[i])) / float(b[i])
        else:
            x = x + abs(float(a[i]) - float(b[i]))
    x=1-x/count

    return x

count=0
our_files=[]
topic=[]
if True:
    path1=args.sum_path
    getfile_story(path1, our_files)
    for file_name in our_files:
        count += 1

        name = re.sub('.*/', '', file_name)
        name = re.sub('\.txt', '', name)
        name = re.sub('\.sum', '', name)

        path =args.re_path
        file = [file_name, path + name + '.npy', ]

        # try:
        a = get_relevance(file)
        topic.append(a)

        if count % 1 == 0:
            print('count: ', count)
            print('topic_relevence\t', np.mean(topic))


    print('topic_relevence\t','\t' ,np.mean(topic))