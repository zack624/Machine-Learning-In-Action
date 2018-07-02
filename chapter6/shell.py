# -*- coding: utf-8 -*-

from numpy import *
import svmMLiA
dm,ls = svmMLiA.loadDataSet('testSet.txt')
b,alphas = svmMLiA.smoSimple(dm,ls,0.6,0.001,40) # 简版smo算法
alphas[alphas>0]
shape(alphas[alphas>0])
svMat = []
for i in range(100):
    if alphas[i]>0.0:
        svMat.append(dm[i])
        print dm[i],ls[i] # 支持向量
svmMLiA.plot(dm,ls,svMat)