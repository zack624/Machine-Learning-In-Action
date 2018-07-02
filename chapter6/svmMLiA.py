# -*- coding: utf-8 -*-

from numpy import *

# 加载数据
def loadDataSet(filename):
    dataMat = []; labels = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        labels.append(float(lineArr[2]))
    return dataMat,labels

# 生成0~m之间且不等于i的随机数
def selectJrand(i,m):
    j = i
    while (j==i):
        j = int(random.uniform(0,m))
    return j

# 保证aj的范围是L~H
def clipAlpha(aj,H,L):
    if aj > H:
        aj = H
    if aj < L:
        aj = L
    return aj

# 简版SMO算法
def smoSimple(dataMatIn,classLabels,C,toler,maxIter):
    dataMatrix = mat(dataMatIn); labelMat = mat(classLabels).transpose()
    b = 0; m,n = shape(dataMatrix)
    alphas = mat(zeros((m,1)))
    iter = 0
    while(iter < maxIter):
        alphaPairsChanged = 0
        for i in range(m):
            fXi = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T)) + b # 预测值
            Ei = fXi - float(labelMat[i]) # 误差值
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)): # i可优化
                j = selectJrand(i,m)
                fXj = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy(); alphaJold = alphas[j].copy();
                if (labelMat[i] != labelMat[j]):
                    L = max(0,alphas[j] - alphas[i])
                    H = min(C,C+alphas[j] - alphas[i])
                else:
                    L = max(0,alphas[j] + alphas[i] - C)
                    H = min(C,alphas[j] + alphas[i])
                if L==H: print "L==H"; continue
                eta =  2.0 * dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T - dataMatrix[j,:]*dataMatrix[j,:].T
                if eta >= 0: print "eta>=0"; continue
                alphas[j] -= labelMat[j]*(Ei - Ej)/eta
                alphas[j] = clipAlpha(alphas[j],H,L)
                if (abs(alphas[j] - alphaJold) < 0.00001): print "j not moving enough"; continue
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])
                b1 = b - Ei - labelMat[i]*(alphas[i] - alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b - Ej - labelMat[i]*(alphas[i] - alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
                if (0 < alphas[i]) and (C > alphas[i]): 
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]): 
                    b = b2
                else: 
                    b = (b1 + b2)/2.0
                alphaPairsChanged += 1
                print "iter: %d i:%d,pairs changed %d" % (iter,i,alphaPairsChanged)
        if (alphaPairsChanged==0): # i不可优化时，迭代数加1
            iter += 1
        else: 
            iter=0
        print "iteration number: %d" % iter
    return b,alphas

# 数据点绘制
def plot(dataMat,labelMat,svMat):
    import matplotlib.pyplot as plt
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    svArr = array(svMat)
    svXcord = list(svArr[:,0]); svYcord = list(svArr[:,1])
    for i in range(n):
        if labelMat[i] == -1.0: xcord1.append(dataArr[i][0]); ycord1.append(dataArr[i][1])
        else: xcord2.append(dataArr[i][0]); ycord2.append(dataArr[i][1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    ax.scatter(svXcord,svYcord,s=30,c='blue')
    plt.xlabel('X1'); plt.ylabel('X2')
    plt.show()


class optStruct:
    def __init__(self,dataMatIn,classLabels,C,toler):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m,1)))
        self.b = 0
        self.eCache = mat(zeros((self.m,2)))


def calcEk(oS,k):
    fXk = float(multiply(oS.alphas,oS.labelMat).T*(oS.X*oS.X[k,].T)) + oS.b
    Ek = fXk - float(oS.labelMat[k])
    return Ek


def selectJ(i,oS,Ei):
    maxK = -1; maxDeltaE = 0; Ej = 0
    oS.eCache[i] = [1,Ei]
    validEcacheList = nonzeros(oS.eCache[:,0].A)[0]
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:
            if k == i: continue
            Ek = calcEk(oS,k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k; maxDeltaE = deltaE; Ej = Ek
        return maxK,Ej
    else:
        j = selectJrand(i,oS.m)
        Ej = calcEk(oS,j)
    return j,Ej


def updateEk(oS,k):
    Ek = calcEk(oS,k)
    oS.eCache[k] = [1,Ek]