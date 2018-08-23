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

my_mat = mat(regTrees.load_data_set('exp2.txt'))
regTrees.create_tree(my_mat, regTrees.model_leaf, regTrees.model_err, (1,10))

train = mat(regTrees.load_data_set('bikeSpeedVsIq_train.txt'))
test = mat(regTrees.load_data_set('bikeSpeedVsIq_test.txt'))
tree = regTrees.create_tree(train, ops=(1, 20))  # regression tree
# tree = regTrees.create_tree(train, regTrees.model_leaf, regTrees.model_err, (1,20))  # model tree
yhat = regTrees.create_forecast(tree, test[:, 0], regTrees.model_tree_eval)
corrcoef(yhat, test[:, 1], rowvar=0)[0, 1]

ws, x, y = regTrees.linear_solve(train)
for i in range(shape(test)[0]):
    yhat[i] = test[i, 0] * ws[1, 0] + ws[0, 0]
corrcoef(yhat, test[:, 1], rowvar=0)[0, 1]


########

import regTrees
from numpy import *
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
my_dat = mat(regTrees.load_data_set('ex00.txt'))
ax.scatter(my_dat[:,0].T.tolist()[0], my_dat[:,-1].T.tolist()[0], s=30)
plt.xlabel('X'); plt.ylabel('Y')
plt.show()


########

from Tkinter import *
root = Tk()
my_label = Label(root, text="Hello World!")
my_label.grid()
root.mainloop()