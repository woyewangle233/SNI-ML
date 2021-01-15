import re
from sumeval.metrics.rouge import RougeCalculator
import os
import numpy as np
from config.config import args
def getfile_story(dir,Filelist):
    if os.path.isfile(dir):
        if True :
            Filelist.append(dir)
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newDir = os.path.join(dir, s)
            getfile_story(newDir,Filelist)
    return Filelist


def find(summary_files,references_files):
    files=[]
    for filename in summary_files:
        id = re.sub('.*\\\\', '', filename)
        id = re.sub('\.txt', '', id)
        id = re.sub('\.story', '', id)
        id = re.sub('\.sum', '', id)

        for f2 in references_files:
            id2 = re.sub('.*\\\\', '', f2)
            id2 = re.sub('\.txt', '', id2)
            id2 = re.sub('\.story', '', id2)
            id2 = re.sub('\.sum', '', id2)

            if id==id2:
                files.append((filename,f2))

    return files


def get_rouge(file):
    f1 = open(file[0], 'r', encoding='utf-8')
    f2 = open(file[1], 'r', encoding='utf-8')
    summary=''
    references=''
    for line in f1.readlines():
        summary+=line
    for line in f2.readlines():
        references+=line
    f1.close()
    f2.close()
    rouge = RougeCalculator(stopwords=True, lang="en")

    rouge_1 = rouge.rouge_n(
        summary,
        references,
        n=1)
    rouge_2 = rouge.rouge_n(
        summary,
        references,
        n=2)
    rouge_l = rouge.rouge_l(
        summary,
        references)
    return rouge_1,rouge_2,rouge_l


if True:
    rouge1 = []
    rouge2 = []
    rougel = []
    count = 0

    our_files = []
    path1 =args.sum_path
    getfile_story(path1, our_files)
    for file_name in our_files:
        count += 1
        name = re.sub('.*/', '', file_name)
        name = re.sub('\.txt', '', name)
        name = re.sub('\.sum', '', name)

        path=args.gold_sum_path
        file=[file_name,path+'/'+name+'.sum',]

        a, b, c = get_rouge(file)
        rouge1.append(a)
        rouge2.append(b)
        rougel.append(c)


    print('rouge1,rouge2,rougel\t', np.mean(rouge1), np.mean(rouge2), np.mean(rougel))

