# -*- coding: utf-8 -*-

from numpy import *

def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().strip().split('\t')) - 1
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

# 线性回归求回归系数
def standRegres(xArr,yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).transpose()
    xTx = xMat.T*xMat
    if linalg.det(xTx) == 0: # 判断xTx的行列式是否为0，即是否可逆
        print "This matrix is singular, cannot do inverse"
        return
    ws = xTx.I*(xMat.T*yMat)
    return ws

# 局部加权线性回归
def lwlr(testPoint,xArr,yArr,k=1.0):
    xMat = mat(xArr); yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye((m))) # 高斯核
    for j in range(m):
        diffMat = testPoint - xMat[j,:]
        weights[j,j] = exp(diffMat*diffMat.T/(-2.0*k**2))
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0:
        print "This matrix is singular, cannot do inverse"
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws


def lwlrTest(testArr,xArr,yArr,k=1.0):
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    return yHat


def rssError(yArr,yHatArr):
    return ((yArr - yHatArr)**2).sum()