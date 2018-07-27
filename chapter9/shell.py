# -*- coding: utf-8 -*-

import regTrees
from numpy import *
testmat = mat(eye(4))
testmat
mat0, mat1 = regTrees.bin_split_data_set(testmat, 1, 0.5)
mat0
mat1
