# -*- coding: utf-8 -*-

import apriori
dataset = apriori.load_data_set()
c1 = apriori.create_c1(dataset)
d = map(set, dataset)
ret, support_data = apriori.scan_d(d, c1, 0.5)

import apriori
dataset = apriori.load_data_set()
L, sd = apriori.find_freq_set(dataset, 0.5)
L, sd = apriori.find_freq_set(dataset, 0.7)

import apriori
dataset = apriori.load_data_set()
L, sd = apriori.find_freq_set(dataset, 0.5)
rules = apriori.generate_rules(L, sd, 0.7)

import apriori
dataset = [line.split() for line in open('chapter11/mushroom.dat').readlines()]
L, sd = apriori.find_freq_set(dataset, 0.3)
for itemset in L[1]:
    if itemset.intersection('2'):
        print itemset