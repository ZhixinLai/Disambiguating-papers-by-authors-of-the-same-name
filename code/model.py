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


# 比较方式一：最短字符串的单词所有都在长字符串都能找到（包括缩写与单词匹配）
# compare two address whether or not is the same by comparing words in the setence ; attention if short word is contained in the longer word, we supposed the words are the same
# input org1 and org2 are organization of two papers "str"
# output is 0or1 and 1 means has the same org
def org_group(org1,org2):
    kkk=1
    if (len(org1)==0 or len(org2)==0):
        kkk=0
    org1=org1.split(" ")
    org2=org2.split(" ")
    org1_len = len(org1)
    org2_len = len(org2)
    org_sim=0
    kk=0
    for i in range(len(org1)):
        for j in range(len(org2)):
            if org1[i] == org2[j]:
                org_sim = org_sim+1
            else:
                if len(org1[i]) != len(org2[j]):
                    if len(org1[i]) > len(org2[j]):
                        short = org2[j]
                        long = org1[i]
                    else:
                        short = org1[i]
                        long = org2[j]
                    for k in range(len(long)-len(short)):
                        long_temp = long[k:k+len(short)]
                        if long_temp == short:
                            org_sim = org_sim+1
    org_sim_ratio = org_sim/min(len(org1), len(org2))
    if org_sim_ratio == 1:
        kk = 1 and kkk
    return kk


# 比较方式二：最短字符串的单词所有都在长字符串都能找到
# compare two address whether or not is the same by comparing words in the setence ; attention if short word is contained in the longer word, we supposed the words are the same
# input org1 and org2 are organization of two papers "str"
# output is 0or1 and 1 means has the same org
def org_group(org1, org2):
    kkk = 1
    if (len(org1) == 0 or len(org2) == 0):
        kkk = 0
    org1 = org1.split(" ")
    org2 = org2.split(" ")
    org1 = list(set(org1))
    org2 = list(set(org2))
    org1_len = len(org1)
    org2_len = len(org2)
    org_sim=0
    kk=0
    for i in range(len(org1)):
        for j in range(len(org2)):
            if org1[i] == org2[j]:
                org_sim = org_sim+1
                break
    org_sim_ratio = org_sim/min(len(org1), len(org2))
    if org_sim_ratio == 1:
        if kkk == 1:
            kk = 1
    return kk

# 比较方式三：两个字符串一模一样（除去一些高频单词之后）########
# compare two address whether or not is the same by comparing words in the setence ; attention if short word is contained in the longer word, we supposed the words are the same
# input org1 and org2 are organization of two papers "str"
# output is 0or1 and 1 means has the same org
def org_group(org1, org2):
    k = 0
    kk = 0
    org1 = org1.split(" ")
    org2 = org2.split(" ")
    if '' in org1:
        org1.remove('')
    if '' in org2:
        org2.remove('')
    if (len(org1) != 0 and len(org2) != 0 and len(org1) == len(org2)):
        for i in range(len(org1)):
            if org1[i] == org2[i]:
                kk = kk+1
        if kk == len(org1):
            k = 1
    return k

# 将不同的 org 元素的相关性转化为矩阵
# 输入是某一个需要进行比较的list-dic(includ key-'org')
# 输出是一个矩阵

def group2max(train_group):
    train_group_dimen = np.zeros((len(train_group), len(train_group)))
    for i in range(len(train_group)):
        for j in range(len(train_group)):
            if org_group(train_group[i]['org'], train_group[j]['org']) == 1:
                train_group_dimen[i][j] = 1
            else:
                train_group_dimen[i][j] = 0
    for i in range(len(train_group)):
        train_group_dimen[i][i] = 1
    return train_group_dimen

# 将不同的 author 元素的相关性转化为矩阵
# 输入是某一个需要进行比较的list(author)
# 输出是一个矩阵

def group3max(train_group, sim_num):
    train_group_dimen = np.zeros((len(train_group), len(train_group)))
    for i in range(len(train_group)):
        for j in range(len(train_group)):
            if coau_group(train_group[i], train_group[j], sim_num) == 1:
                train_group_dimen[i][j] = 1
            else:
                train_group_dimen[i][j] = 0
    for i in range(len(train_group)):
        train_group_dimen[i][i] = 1
    return train_group_dimen


#利用一个矩阵，将相同元素抓在一个一个集合，进而得到一个list
#input 是一个矩阵， output是一个list

def max2list(train_group_dimen):
    same_autorg=[] # 放置某一组论文——共作
    forever=1
    for i in range(train_group_dimen.shape[1]):
        same_autorg_stru = [] # 放置某一组论文——共作+共org
        same_autorg_stru_temp = [] # 临时放置某一组论文——共作+共org
        differ = []
        same_autorg_stru.append(i)
        same_autorg_stru_temp.append(i)
        differ.append(i)
        p=0
        for t in same_autorg:
            if i in t:
                p = p+1
        if p != 0:
            continue
        while forever > 0:
            same_autorg_stru_temp = copy.deepcopy(differ)
            for q in differ:
                for j in range(train_group_dimen.shape[1]):
                    if train_group_dimen[q][j] == 1:
                        same_autorg_stru_temp.append(j)
                if q == differ[-1]:
                    same_autorg_stru_temp = list(set(same_autorg_stru_temp)) # 去掉重复元素
                    same_autorg_stru_temp.sort() # 从小到大排序
                    differ = [k for k in same_autorg_stru_temp if k not in same_autorg_stru]
                    same_autorg_stru.extend(differ)

                    same_autorg_stru.sort()

            if len(differ) == 0:
                break
        same_autorg.append(same_autorg_stru)
    return same_autorg


# 定义class 用于doc2vect 转化
class LabeledLineSentence(object):
    def __init__(self, sources):
        self.sources = sources
        flipped = {}

        # make sure that keys are unique
        for key, value in sources.items():
            if value not in flipped:
                flipped[value] = [key]
            else:
                raise Exception('Non-unique prefix encountered')

    def __iter__(self):
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    yield LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no])

    def to_array(self):
        self.sentences = []
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    self.sentences.append(LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
        return self.sentences

    def sentences_perm(self):
        shuffle(self.sentences)
        return self.sentences


# 计算向量余弦
def cosVector(x, y):
    if (len(x) != len(y)):
        print('error input,x and y is not in the same space')
        return
    result1 = 0.0
    result2 = 0.0
    result3 = 0.0
    for i in range(len(x)):
        result1 += x[i] * y[i]  # sum(X*Y)
        result2 += x[i] ** 2  # sum(X*X)
        result3 += y[i] ** 2  # sum(Y*Y)
    result = result1 / ((result2 * result3) ** 0.5)
    return result

