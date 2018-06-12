# -*- coding: utf-8 -*-

from numpy import *
import operator
from os import listdir

# kNN小demo数据
def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    classLabelVector = ['A','A','B','B']
    return group,classLabelVector

# kNN核心的分类算法
def classify0(inX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX,(dataSetSize,1)) - dataSet # tile()将inX:[0,0]扩展成有dataSetSize行的矩阵
    sqDiffMat = diffMat**2 # 矩阵各值的2次幂
    sqDistances = sqDiffMat.sum(axis=1) # 矩阵每行(axis=1)的值的总和
    distances = sqDistances**0.5 # 矩阵各值的平方
    sortedDistIndicies = distances.argsort() # distances从小到大的索引值位置，即distances[sortedDistIndicies[0]]为最小值
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

# 从datingTestSet2.txt中读取训练数据
def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines,3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector

# 归一化数据，将(所有数值-min)/ranges，范围为0-1
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals,(m,1))
    normDataSet = normDataSet/tile(ranges,(m,1))
    return normDataSet,ranges,minVals

# 测试算法：取部分数据作为测试数据，比较得出错误率
def datingClassTest():
    hoRatio = 0.10
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    normMat,ranges,minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3) # 训练数据取测试数据以外的全部数据
        print "the classifier came back with: %d,the real answer is: %d" % (classifierResult,datingLabels[i])
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
    return "the total error rate is: %f" % (errorCount/float(numTestVecs))

# 实际应用：对任一数据进行分类判断
def classifyPerson():
    resultList = ["not at all","in small doses","in large doses"]
    a = float(raw_input("frequent flier miles earned per year?"))
    b = float(raw_input("percentage of time spent playing video games?"))
    c = float(raw_input("liters of ice cream consumed per year?"))
    datingDataMat,datingLabels = file2matrix("datingTestSet2.txt")
    normMat,ranges,minVals = autoNorm(datingDataMat)
    inArr = array([a,b,c])
    result = classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
    return "You will probably like this person: " + resultList[result - 1]

# 数字图像转化成矩阵数据
def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        line = fr.readline()
        for j in range(32):
            returnVect[0,i*32+j] = int(line[j]) # 将一个32*32的图像转化成1*1024的数据
    return returnVect

# 测试手写数字识别算法：trainingDigits作为训练数据，testDigits作为测试数据
def handwritingClassTest():
    files = listdir("trainingDigits")
    fileCount = len(files)
    trainingMat = zeros((fileCount,1024))
    hWLabels = []
    for i in range(fileCount):
        fileName = files[i].split(".")[0]
        trainingMat[i,:] = img2vector("trainingDigits/%s" % files[i])
        digit = int(fileName.split("_")[0])
        hWLabels.append(digit)
    testFiles = listdir("testDigits")
    testFileCount = len(testFiles)
    errorCount = 0.0
    for i in range(testFileCount):
        realDigit = int(testFiles[i].split(".")[0].split("_")[0])
        rs = classify0(img2vector("testDigits/%s" % testFiles[i]),trainingMat,hWLabels,3)
        print "predict: %d,real number: %d" % (rs,realDigit)
        if (rs != realDigit): 
            errorCount += 1.0
    print "\nthe total number of error is: %d" % errorCount
    print "\nthe total error rate is: %f" % (errorCount/float(testFileCount))