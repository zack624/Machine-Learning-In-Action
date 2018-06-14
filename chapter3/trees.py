# -*- coding: utf-8 -*-

'''
ID3算法
'''

from math import log
import operator

# 计算香农熵: H = -∑p(xi)*log2(p(xi))
def calcShannonEnt(dataSet):
    m = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        label = featVec[-1] # 指定标签列
        if label not in labelCounts.keys():
            labelCounts[label] = 0
        labelCounts[label] += 1 # 统计标签列各值出现次数
    shannonEnt = 0.0
    for j in labelCounts:
        prob = float(labelCounts[j])/m # p(xi)
        shannonEnt -= prob * log(prob,2)
    return shannonEnt

# 构造数据集
def createDataSet():
    dataSet = [[1,1,'yes'],
            [1,1,'yes'],
            [1,0,'no'],
            [0,1,'no'],
            [0,1,'no']]
    labels = ["no surfacing","flippers"] # 除分类列外的所有列名
    return dataSet,labels

# 根据轴axis==value划分出所需数据集
def splitDataSet(dataSet,axis,value):
    retDataSet = []
    for featVec in dataSet:
        if (featVec[axis] == value):
            reduceFeatVec = featVec[:axis]
            reduceFeatVec.extend(featVec[axis+1:]) # extend()去除原list，直接合并成一个list
            retDataSet.append(reduceFeatVec)
    return retDataSet

# 选出熵最大的列
def chooseBestFeatureToSplit(dataSet):
    colNum = len(dataSet[0]) - 1 # 去掉分类列外的所有列的数量
    baseEnt = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(colNum):
        featVals = [example[i] for example in dataSet] # 当前列的所有可取值，需去重
        uniqueFeatVals = set(featVals)
        newEnt = 0.0
        for feat in uniqueFeatVals:
            sDataSet = splitDataSet(dataSet,i,feat) # 对当前列的当前值进行划分，取出子数据集
            prob = float(len(sDataSet))/float(len(dataSet))
            newEnt += prob * calcShannonEnt(sDataSet)
        infoGain = baseEnt - newEnt
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

# 取出list中出现最多次的值
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

# 递归创建决策树
def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet] # 分类列的所有可取值
    if (classList.count(classList[0]) == len(dataSet)): # 只剩一种分类，直接返回类值
        return classList[0]
    if (len(dataSet[0]) == 1): # 遍历完所有列/特征，返回分类列下出现次数最多的类值
        return majorityCnt(classList)
    bestFeature = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeature]
    myTree = {bestFeatLabel:{}} # 构造决策树：{列名：{列值：所属类别,列值：{...}}}
    del(labels[bestFeature]) # 去除labels中已用的列
    bestFeatValues = [example[bestFeature] for example in dataSet]
    uniqueBestFeatVals = set(bestFeatValues)
    for uf in uniqueBestFeatVals:
        subLabels = labels[:]
        sds = splitDataSet(dataSet,bestFeature,uf)
        myTree[bestFeatLabel][uf] = createTree(sds,subLabels)
    return myTree

# 递归查找决策树的相应叶子结点，完成分类
def classify(inTree,labels,testVec):
    root = inTree.keys()[0]
    subTree = inTree[root]
    featIndex = labels.index(root)
    testVal = testVec[featIndex]
    if type(subTree[testVal]).__name__ == 'dict': # 如果有子树，则递归查找
        classLabel = classify(subTree[testVal],labels,testVec)
    else: # 如果是叶子结点，则为所求分类
        classLabel = subTree[testVal]
    return classLabel

# 序列化dict
def storeTree(inTree,filename):
    import pickle
    fw = open(filename,'w')
    pickle.dump(inTree,fw)
    fw.close()

# 反序列化dict
def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)