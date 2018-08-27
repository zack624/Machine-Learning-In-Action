# -*- coding: utf-8 -*-

from numpy import *
import Kmeans
datmat = mat(Kmeans.load_data_set('testSet.txt'))
# Kmeans.dist_eclud(datmat[0], datmat[1])
# Kmeans.rand_cent(datmat, 2)
cens, censa = Kmeans.kmeans(datmat, 4)

datmat2 = mat(Kmeans.load_data_set('testSet2.txt'))
cens, censa = Kmeans.bikmeans(datmat2, 3)

########

import Kmeans
from numpy import *
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
my_dat = mat(Kmeans.load_data_set('testSet.txt'))
ax.scatter(my_dat[:, 0].T.tolist()[0], my_dat[:, -1].T.tolist()[0], s=30)
plt.xlabel('X'); plt.ylabel('Y')
plt.show()