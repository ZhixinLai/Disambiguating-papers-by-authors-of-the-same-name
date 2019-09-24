"""
@author: zhixin lai
@contact: laizhixin16@gmail.com
"""
import os
import re as re
import json
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
from sklearn.model_selection import train_test_split
import string
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution1D, MaxPooling1D
from keras.utils import np_utils
from keras import backend as K
import copy
import Levenshtein
from collections import Counter
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
import numpy
from random import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics
from data_preprocess import *
from model import *


def data_load(data_name):

    if data_name == "pubs_train":
        f = open("../data/pubs_train.json")
    elif data_name == "pubs_train":
        f = open("../data/assignment_train.json")
    elif data_name == "pubs_train":
        f = open("../data/pubs_validate.json")
    elif data_name == "pubs_train":
        f = open("../data/assignment_validate.json")
    elif data_name == "pubs_train":
        f = open("../data/pubs_test.json")
    val_pub = json.load(f)
    f.close

    return val_pub


def __main__():

    submit = {}
    for alllop in tqdm(val_pub):

        # step0:抓出某个同名作者的所有论文，组成list，信息包括论文的编号、机构、共同作者、关键词,并且将所有作者的名字和lima所在机构的名称预处理
        # 输出的aut_samename是只有lima作者的所有闲相关信息：论文的编号、机构、共同作者、关键词
        aut_samename = []
        for i in val_pub[alllop]:
            name = name_prepro(alllop)
            aut_samename_temp = {'id': '5bc6bfde486cef66309f115a',
                                 'org': 'Changchun Inst. of Applied Chemistry',
                                 'keywords': ['C60', 'Ion-molecule reaction', 'Methyl ethers']}
            aut_samename_temp['id'] = copy.deepcopy(i['id'])
            aut_samename_temp['authors'] = copy.deepcopy(i['authors'])
            aut_samename_temp['keywords'] = copy.deepcopy(i['keywords'])
            aut_samename_temp['title'] = copy.deepcopy(i['title'])

            for j in i['authors']:
                k = name_prepro(j['name'])
                if k == name:
                    aut_samename_temp['org'] = copy.deepcopy(j['org'])
            aut_samename.append(aut_samename_temp)

        for i in aut_samename:
            for j in i['authors']:
                j['name'] = name_prepro(j['name'])
            for j in i['keywords']:
                j = keywords_prepro(j)
            i['org'] = org_prepro(i['org'])
            i['title'] = title_prepro(i['title'])

        # step1：对lima作者的论文进行机构合并
        orgsim_max = group2max(aut_samename)
        orgsim_list = max2list(orgsim_max)



        # step2：对有共同作者的group进行名字合并
        autnam_list = []
        for i in orgsim_list:
            autnam_list_temp = []
            for j in i:
                for k in aut_samename[j]['authors']:
                    autnam_list_temp.append(k['name'])
            autnam_list_temp = list(set(autnam_list_temp))
            autnam_list.append(autnam_list_temp)


        # step3: 对step1中的集合根据作者名字再进行合并

        autsim_max = group3max(autnam_list, 3)
        autsim_list = max2list(autsim_max)
        aut_name_list = []
        for i in autsim_list:
            temp = []
            for j in i:
                temp.extend(orgsim_list[j])
            temp.sort()
            aut_name_list.append(temp)


        # step4：有共同作者和共同机构的group进行关键词合并
        keywords_list = []
        for i in aut_name_list:
            keywords_list_temp = []
            for j in i:
                keywords_list_temp.extend(aut_samename[j]['keywords'])
            keywords_list_temp = list(set(keywords_list_temp))
            if '' in keywords_list_temp:
                keywords_list_temp.remove('')
            keywords_list.append(keywords_list_temp)


        # step5：对step3中的集合根据关键词再进行合并
        # 5.1 得到基于keywords_list的矩阵

        worsim_max = group3max(keywords_list, 2)
        worsim_list = max2list(worsim_max)

        # 5.2 转化为目的矩阵
        aut_name_word_list = []
        for i in worsim_list:
            temp = []
            for j in i:
                temp.extend(aut_name_list[j])
            temp.sort()
            aut_name_word_list.append(temp)
        print('len(aut_name_word_list):', len(aut_name_word_list))

        # step6 做词典处理
        # 6.1 将所有title取出，用于训练
        title_all = []
        for i in aut_samename:
            title_all.append(i['title'])

        # 6.2文件保存、读取和删除
        f = open("../train_title.txt", "w")
        for list_mem in title_all:
            f.write(list_mem + "\n")
        f.close

        f = open("../train_title.txt", "r")
        content = f.readlines()
        f.close

        contend_length = len(content)  # 用于网络添加训练参数

        # 6.3 利用class将文本转化为可以输入的格式
        sources = {'../train_title.txt': 'train_title'}
        sentences = LabeledLineSentence(sources)

        # 6.4构建Doc2vec模型
        model = Doc2Vec(min_count=1, window=4, size=100, sample=1e-4, negative=5, workers=8)
        model.build_vocab(sentences.to_array())

        # 6.5训练Doc2vec模型（本例迭代次数为10，如果时间允许，可以迭代更多的次数）
        for epoch in tqdm(range(5)):
            model.train(sentences.sentences_perm(), total_examples=contend_length, epochs=1)

        # 6.6将训练好的句子存放在train_arrays中，用于后文分类使用
        train_arrays = numpy.zeros((contend_length, 100))
        for i in range(contend_length):
            prefix_train_neg = 'train_title_' + str(i)
            train_arrays[i] = model.docvecs[prefix_train_neg]
        print(train_arrays.shape)

        # 6.7删除文件，防止下一次循环，内容无法完全覆盖上一次内容
        os.remove("../train_title.txt")

        # step7：对step5中的集合，对于剩余的单个作者合并到前面的集合
        temp = []  # 之后直接往temp中加
        for i in aut_name_word_list:
            if len(i) > 1:
                temp.append(i)

        aut_name_word_title_list = copy.deepcopy(temp)

        for i in aut_name_word_list:
            if len(i) == 1:
                dic_temp = {}
                for j in aut_name_word_title_list:
                    for k in j:
                        dic_temp[k] = cosVector(train_arrays[i[0]], train_arrays[k])
                max_key = max(dic_temp, key=dic_temp.get)
                for p in aut_name_word_title_list:
                    if max_key in p:
                        p.append(i[0])
                        p.sort()

        # step8：将序号对应的文章序列存入
        final = []
        for i in aut_name_word_title_list:
            temple = []
            for j in i:
                temple.append(aut_samename[j]['id'])
            final.append(temple)
        submit[alllop] = final

    f = open("../data/assignment_validate.json", "w")
    json.dump(submit, f, ensure_ascii=False)
    f.close