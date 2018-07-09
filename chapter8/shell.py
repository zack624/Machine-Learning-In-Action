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