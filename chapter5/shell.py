# -*- coding: utf-8 -*-

from numpy import *
import logRegres
data,ls = logRegres.loadDataSet()
wei1 = logRegres.gradAscent(data,ls)
logRegres.plotBestFit(wei1)


reload(logRegres)
wei2 = logRegres.stocGradAscent0(array(data),ls)
logRegres.plotBestFit(wei2)


wei3 = logRegres.stocGradAscent1(array(data),ls)
logRegres.plotBestFit(wei3)