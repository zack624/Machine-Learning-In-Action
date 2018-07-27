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


def bin_split_data_set(dataset, feature, value):
    mat0 = dataset[nonzero(dataset[:, feature] > value)[0], :][0]
    mat1 = dataset[nonzero(dataset[:, feature] <=  )[0], :][0]
    return mat0, mat1


# def create_tree(dataset, leaf_type=reg_leaf, err_type=reg_err, ops=(1,4)):
#     feat, val = choose_best_split(dataset, leaf_type, err_type, ops)
#     if feat is None: return val
#     ret_tree = {}
#     ret_tree['sp_ind'] = feat
#     ret_tree['sp_val'] = val
#     lset, rset = bin_split_data_set(dataset, feat, val)
#     ret_tree['left'] = create_tree(lset, leaf_type, err_type, ops)
#     ret_tree['right'] = create_tree(rset, leaf_type, err_type, ops)
#     return ret_tree








