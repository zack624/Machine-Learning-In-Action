# -*- coding: utf-8 -*-

from numpy import *
import bayes
postList,classVec = bayes.loadDataSet()
vocabList = bayes.createVocabList(postList)
mat = zeros((len(postList),len(vocabList)))
for i in range(len(postList)):
    mat[i] = bayes.setOfWords2Vec(vocabList,postList[i])
p0,p1,pAbusive = bayes.trainNB0(mat,classVec)

