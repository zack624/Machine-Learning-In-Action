# -*- coding: utf-8 -*-

import regTrees
from numpy import *
testmat = mat(eye(4))
mat0, mat1 = regTrees.bin_split_data_set(testmat, 1, 0.5)

my_dat1 = regTrees.load_data_set('ex00.txt')
tree = regTrees.create_tree(mat(my_dat1))

my_dat1 = regTrees.load_data_set('ex2.txt')
tree = regTrees.create_tree(mat(my_dat1), ops=(0, 1))
print(tree)
my_test1 = regTrees.load_data_set('ex2test.txt')
t = regTrees.prune(tree, mat(my_test1))
print(t)