# -*- coding: utf-8 -*-

from numpy import *


def load_data_set(filename):
    fr = open(filename)
    datamat = []
    for line in fr.readlines():
        curline = line.strip().split('\t')
        fltline = map(float, curline)
        datamat.append(fltline)
    return datamat


# 对数据集进行二切分
def bin_split_data_set(dataset, feature, value):
    # nonzero()返回一个元组，元组第一项为各非零值的x轴，第二项为各非零值的y轴
    mat0 = dataset[nonzero(dataset[:, feature] > value)[0], :]
    mat1 = dataset[nonzero(dataset[:, feature] <= value)[0], :]
    return mat0, mat1


def reg_leaf(dataset):
    return mean(dataset[:, -1])


# 总方差
def reg_err(dataset):
    return var(dataset[:, -1]) * shape(dataset)[0]


def create_tree(dataset, leaf_type=reg_leaf, err_type=reg_err, ops=(1,4)):
    feat, val = choose_best_split(dataset, leaf_type, err_type, ops)
    if feat is None: return val
    ret_tree = {}
    ret_tree['sp_ind'] = feat
    ret_tree['sp_val'] = val
    lset, rset = bin_split_data_set(dataset, feat, val)
    ret_tree['left'] = create_tree(lset, leaf_type, err_type, ops)
    ret_tree['right'] = create_tree(rset, leaf_type, err_type, ops)
    return ret_tree


def choose_best_split(dataset, leaf_type=reg_leaf, err_type=reg_err, ops=(1,4)):
    tols = ops[0]; toln = ops[1]  # tols误差下降值,toln切分最少样本数，修改可以实现预剪枝
    if len(set(dataset[:, -1].T.tolist()[0])) == 1:
        return None, leaf_type(dataset)
    m, n = shape(dataset)
    S = err_type(dataset)
    bests = inf; best_index = 0; best_value = 0
    for feat_index in range(n-1):
        for split_value in set(dataset[:, feat_index].T.tolist()[0]):
            mat0, mat1 = bin_split_data_set(dataset, feat_index, split_value)
            if (shape(mat0)[0] < toln) or (shape(mat1)[0] < toln): continue
            news = err_type(mat0) + err_type(mat1)
            if news < bests:
                bests = news
                best_index = feat_index
                best_value = split_value
    if (S - bests) < tols:
        return None, leaf_type(dataset)
    mat0, mat1 = bin_split_data_set(dataset, best_index, best_value)
    if (shape(mat0)[0] < toln) or (shape(mat1)[0] < toln):
        return None, leaf_type(dataset)
    return best_index, best_value


def is_tree(obj):
    return type(obj).__name__ == 'dict'


def get_mean(tree):
    if is_tree(tree['right']): tree['right'] = get_mean(tree['right'])
    if is_tree(tree['left']): tree['left'] = get_mean(tree['left'])
    return (tree['left'] + tree['right'])/2.0


def prune(tree, test_data):
    if shape(test_data) == 0: return get_mean(tree)
    if is_tree(tree['right']) or is_tree(tree['left']):
        lset, rset = bin_split_data_set(test_data, tree['sp_ind'], tree['sp_val'])
    if is_tree(tree['left']): tree['left'] = prune(tree['left'], lset)
    if is_tree(tree['right']): tree['right'] = prune(tree['right'], rset)
    if not is_tree(tree['right']) and not is_tree(tree['left']):
        lset, rset = bin_split_data_set(test_data, tree['sp_ind'], tree['sp_val'])
        error_no_merge = sum(power(lset[:, -1] - tree['left'], 2)) +\
                         sum(power(rset[:, -1] - tree['right'], 2))
        tree_mean = (tree['left'] + tree['right'])/2.0
        error_merge = sum(power(test_data[:, -1] - tree_mean, 2))
        if error_merge < error_no_merge:
            print "merging"
            return tree_mean
        else:
            return tree
    else: return tree



