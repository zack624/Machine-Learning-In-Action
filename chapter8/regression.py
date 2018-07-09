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