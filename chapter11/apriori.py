# -*- coding: utf-8 -*-


def load_data_set():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]


def create_c1(dataset):
    # generate list c1
    c1 = []
    for tran in dataset:
        for item in tran:
            if not [item] in c1:
                c1.append([item])
    # c1 sort and change items to frozenset
    c1.sort()
    return map(frozenset, c1)


def scan_d(dataset, ck, min_support):
    # use dict to record the support_count of every item in ck
    itemset_count = {}
    for tran in dataset:
        for can in ck:
            if can.issubset(tran):  # set.issubset()
                if can in itemset_count:
                    itemset_count[can] += 1
                else:
                    itemset_count[can] = 1
    # calculate support rate of every item in dict and choose frequent itemset
    tran_num = float(len(dataset))
    ret_list = []
    support_data = {}
    for itemset in itemset_count:
        support = itemset_count[itemset]/tran_num
        if support >= min_support:
            ret_list.insert(0, itemset)
            support_data[itemset] = support
    return ret_list, support_data

