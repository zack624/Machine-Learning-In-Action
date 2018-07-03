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


import svmMLiA
dm,ls = svmMLiA.loadDataSet('testSet.txt')
b,alphas = svmMLiA.smoP(dm,ls,0.6,0.001,40)
ws = svmMLiA.calcWs(alphas,dm,ls)
mat(dm)[0]*mat(ws) + b # 对第0个数据分类
ls[0]


import svmMLiA
dm,ls = svmMLiA.loadDataSet('testSet.txt')
b,alphas = svmMLiA.smoP(dm,ls,0.6,0.001,40)
svMat = []
for i in range(100):
    if alphas[i]>0.0:
        svMat.append(dm[i])

ws = svmMLiA.calcWs(alphas,dm,ls)
svmMLiA.plot(dm,ls,svMat,ws,float(b))


import svmMLiA
dm,ls = svmMLiA.loadDataSet('testSetRBF.txt')
b,alphas = svmMLiA.smoP(dm,ls,200,0.0001,10000,('rbf',1.3))
sv = []
for i in range(100):
    if alphas[i]>0.0:
        sv.append(dm[i])

svmMLiA.plot(dm,ls,sv)


import svmMLiA
svmMLiA.testDigits(('rbf',50))
# fullSet,iter: 11 i 1933,pairs changed 0
# iteration number: 12
# there are 176 Support Vectors
# the training error rate is: 0.000000
# the test error rate is: 0.012685