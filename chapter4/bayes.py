# -*- coding: utf-8 -*-

from numpy import *

def loadDataSet():
    postingList = [['my','dog','has','flea','problems','help','please'],
            ['maybe','not','take','him','to','dog','park','stupid'],
            ['my','dalmation','is','so','cute','I','love','him'],
            ['stop','posting','stupid','worthless','garbage'],
            ['mr','licks','ate','my','steak','how','to','stop','him'],
            ['quit','buying','worthless','dog','food','stupid']
            ]
    classVec = [0,1,0,1,0,1]
    return postingList,classVec

# 创建词汇表
def createVocabList(dataSet):
    vocabList = set([])
    for document in dataSet:
        vocabList = vocabList | set(document)
    return list(vocabList)

# 句子转换成向量：同词汇表长度，单词出现则为1，否则为0
def setOfWords2Vec(vocabList,inWords):
    retList = [0]*len(vocabList)
    for word in inWords:
        if word in vocabList:
            retList[vocabList.index(word)] = 1
        else:
            print "the word: %s is not in my vocaburary." % word
    return retList


def trainNB0(trainMatrix,trainCategory):
    docNum = len(trainMatrix)
    vocabLen = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(len(trainCategory))
    p0Mat = zeros(vocabLen); p1Mat = zeros(vocabLen)
    p0Num = 0.0; p1Num = 0.0
    for i in range(docNum):
        if trainCategory[i] == 0:
            p0Mat += trainMatrix[i]
            p0Num += sum(trainMatrix[i])
        else:
            p1Mat += trainMatrix[i]
            p1Num += sum(trainMatrix[i])
    p0Rate = p0Mat/p0Num
    p1Rate = p1Mat/p1Num
    return p0Rate,p1Rate,pAbusive