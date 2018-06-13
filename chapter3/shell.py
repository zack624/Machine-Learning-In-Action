# -*- coding: utf-8 -*-

import trees
ds,ls = trees.createDataSet()
trees.chooseBestFeatureToSplit(ds)

import trees
ds,ls = trees.createDataSet()
trees.createTree(ds,ls)

import treePlotter
mt = treePlotter.retrieveTree(0)
treePlotter.createPlot(mt)

import trees
import treePlotter
it = treePlotter.retrieveTree(0)
ds,ls = trees.createDataSet()
trees.classify(it,ls,[0,0])

import trees
import treePlotter
it = treePlotter.retrieveTree(0)
trees.storeTree(it,'classifierStorage.txt')
ot = trees.grabTree('classifierStorage.txt')