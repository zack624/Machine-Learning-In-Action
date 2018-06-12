# -*- coding: utf-8 -*-

# kNN小demo测试
import kNN
group,labels = kNN.createDataSet()
kNN.classify0([0,0],group,labels,3)

# 分析数据：matplotlib创建散点图
import kNN
import matplotlib
import matplotlib.pyplot as plt
from numpy import *
datingDataMat,datingLabels = kNN.file2matrix('datingTestSet2.txt')
fig = plt.figure()
ax = fig.add_subplot(111)
# ax.scatter(datingDataMat[:,1],datingDataMat[:,2])
# ax.scatter(datingDataMat[:,1],datingDataMat[:,2],15.0*array(datingLabels),15.0*array(datingLabels))
ax.scatter(datingDataMat[:,0],datingDataMat[:,1],15.0*array(datingLabels),15.0*array(datingLabels)) # 以datingDataMat的0，1列分别作为x,y值，并标注出不同标签
plt.show()

# 取datingTestSet2.txt的训练数据，并归一化数据数值
import kNN
datingDataMat,datingLabels = kNN.file2matrix('datingTestSet2.txt')
normMat,ranges,minVals = kNN.autoNorm(datingDataMat)