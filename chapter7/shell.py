# -*- coding: utf-8 -*-

import adaboost
from numpy import *
da,la = adaboost.loadDataSet('horseColicTraining.txt')
ca = adaboost.adaBoostTrainDS(da,la,10)
tda,tla = adaboost.loadDataSet('horseColicTest.txt')
prediction10 = adaboost.adaClassify(tda,ca)
errArr = mat(ones((67,1)))
errArr[prediction10 != mat(tla).T].sum()