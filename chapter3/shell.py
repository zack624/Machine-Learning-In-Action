# -*- coding: utf-8 -*-

# 求使数据集熵最大的列
import trees
ds,ls = trees.createDataSet()
trees.chooseBestFeatureToSplit(ds)

# 创建决策树
import trees
ds,ls = trees.createDataSet()
trees.createTree(ds,ls)

# 绘制树
import treePlotter
mt = treePlotter.retrieveTree(0)
treePlotter.createPlot(mt)

# 利用决策树判断分类
import trees
import treePlotter
it = treePlotter.retrieveTree(0)
ds,ls = trees.createDataSet()
trees.classify(it,ls,[0,0])

# 序列化与反序列化决策树
import trees
import treePlotter
it = treePlotter.retrieveTree(0)
trees.storeTree(it,'classifierStorage.txt')
ot = trees.grabTree('classifierStorage.txt')

# 隐形眼镜数据集测试
import trees
import treePlotter
fr = open('lenses.txt')
ds = [example.strip().split("\t") for example in fr.readlines()]
ls = ['age','prescript','antigmatic','tearRate']
mt = trees.createTree(ds,ls)
treePlotter.createPlot(mt)