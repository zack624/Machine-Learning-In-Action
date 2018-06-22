# -*- coding: utf-8 -*-

from numpy import *

def loadDataSet():
    fr = open('testSet.txt')
    mat = []; labels = []
    for line in fr.readlines():
        line = line.strip().split()
        mat.append([1.0,float(line[0]),float(line[1])])
        labels.append(int(line[2]))
    return mat,labels


def sigmoid(inX):
    return 1.0/(1+exp(-inX))


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