# -*- coding: utf-8 -*-

import regression
from numpy import *
dm,ls = regression.loadDataSet('ex0.txt')
ws = regression.standRegres(dm,ls)

xMat = mat(dm)
yMat = mat(ls)
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(xMat[:,1].flatten().A[0],yMat.T[:,0].flatten().A[0]) # flatten()将多维矩阵折叠成一维
xCopy = xMat.copy()
xCopy.sort(0)
yHat = xCopy*ws
ax.plot(xCopy[:,1],yHat)
plt.show()

xMat = mat(dm)
yMat = mat(ls)
yHat = xMat*ws
corrcoef(yHat.T,yMat) # 求相关系数

# 局部加权线性回归
import regression
from numpy import *
dm,ls = regression.loadDataSet('ex0.txt')
ls[0]
regression.lwlr(dm[0],dm,ls,1.0)
yHat = regression.lwlrTest(dm,dm,ls,0.01)

xMat = mat(dm)
strInd = xMat[:,1].argsort(0)
xSort = xMat[strInd][:,0,:]
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(xSort[:,1],yHat[strInd])
ax.scatter(xMat[:,1].flatten().A[0],mat(ls).T.flatten().A[0],s=2,c='red')
plt.show()

# 预测鲍鱼寿命
import regression
from numpy import *
abX,abY = regression.loadDataSet('abalone.txt')
yHat01 = regression.lwlrTest(abX[0:99],abX[0:99],abY[0:99],0.1)
yHat1 = regression.lwlrTest(abX[0:99],abX[0:99],abY[0:99],1)
yHat10 = regression.lwlrTest(abX[0:99],abX[0:99],abY[0:99],10)
regression.rssError(abY[0:99],yHat01.T)
regression.rssError(abY[0:99],yHat1.T)
regression.rssError(abY[0:99],yHat10.T)

yHat01 = regression.lwlrTest(abX[100:199],abX[0:99],abY[0:99],0.1)
regression.rssError(abY[100:199],yHat01.T)
yHat1 = regression.lwlrTest(abX[100:199],abX[0:99],abY[0:99],1)
regression.rssError(abY[100:199],yHat1.T)
yHat10 = regression.lwlrTest(abX[100:199],abX[0:99],abY[0:99],10)
regression.rssError(abY[100:199],yHat10.T)

ws = regression.standRegres(abX[0:99],abY[0:99])
yHat = mat(abX[100:199])*ws
regression.rssError(abY[100:199],yHat.T.A)

# 岭回归
import regression
dm,ls = regression.loadDataSet('abalone.txt')
ridgeWeights = regression.ridgeTest(dm,ls)
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(ridgeWeights)
plt.show()

# 前向逐步回归
import regression
from numpy import *
dm,ls = regression.loadDataSet('abalone.txt')
wMat = regression.stageWise(dm,ls,0.01,200)
wMat = regression.stageWise(dm,ls,0.001,5000)
xMat = mat(dm)
yMat = mat(ls).T
xMat = (xMat - mean(xMat,0))/var(xMat,0)
yMat = yMat - mean(yMat,0)
regression.standRegres(xMat,yMat.T).T


import regression
regression.setDataCollect()


import regression
from numpy import *
lgx,lgy = regression.loadLEGOData('out.txt')
# ws = regression.standRegres(lgx,lgy)
# lgx = mat(lgx)
# lgx[0] * ws
ws = regression.crossValidation(lgx,lgy,10)
regression.ridgeTest(lgx,lgy)