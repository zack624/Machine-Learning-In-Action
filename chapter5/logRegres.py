# -*- coding: utf-8 -*-

from numpy import *

# 从testSet.txt中加载数据[x0,x1,x2]
def loadDataSet():
    fr = open('testSet.txt')
    mat = []; labels = []
    for line in fr.readlines():
        line = line.strip().split()
        mat.append([1.0,float(line[0]),float(line[1])])
        labels.append(int(line[2]))
    return mat,labels

# sigmoid函数
def sigmoid(inX):
    return 1.0/(1+exp(-inX))

# 绘制数据点和图像
def plotBestFit(wei):
    import matplotlib.pyplot as plt
    if type(wei).__name__ != 'ndarray':
        weights = wei.getA()
    else:
        weights = wei
    dataMat,labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    x = arange(-3.0,3.0,0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x,y)
    plt.xlabel('X1'); plt.ylabel('X2')
    plt.show()

# w=w+a(y-sigmoid(x))x
def gradAscent(dataMat,labels):
    dMat = mat(dataMat)
    lMat = mat(labels).transpose()
    m,n = shape(dMat)
    alpha = 0.001
    weights = ones((n,1))
    for i in range(500):
        h = sigmoid(dMat*weights)
        error = (lMat - h)
        weights = weights + alpha*dMat.transpose()*error
    return weights


def stocGradAscent0(dataMatrix,classLabels):
    m,n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i] - h
        weights = weights + alpha*error*dataMatrix[i]
    return weights


def stocGradAscent1(dataMatrix,classLabels,numIter=150):
    m,n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.01
            randIndex = int(random.uniform(0,len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha*error*dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights
