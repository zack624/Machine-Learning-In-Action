# -*- coding: utf-8 -*-

import svmMLiA
dm,ls = svmMLiA.loadDataSet('testSetRBF.txt')
b,alphas = svmMLiA.smoSimple(dm,ls,0.6,0.001,40)