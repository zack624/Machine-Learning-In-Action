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


def apriori_gen(Lk, k):
    # use the Lk to generate next Ci by join in two for loops
    Ci = []
    len_of_L = len(Lk)
    for i in range(len_of_L):
        for j in range(i+1, len_of_L):
            items1 = list(Lk[i])[:k-2]
            items2 = list(Lk[j])[:k-2]
            items1.sort()
            items2.sort()
            if items1 == items2:
                cur_item = Lk[i] | Lk[j]
                # apply apriori principle
                is_freq = True
                for x in cur_item:
                    if (cur_item - frozenset([x])) not in Lk:
                        is_freq = False
                        print "apply apriori principle and find:", cur_item
                        break
                if is_freq:
                    Ci.append(cur_item)
    return Ci


def apriori(dataset, min_support=0.5):
    # generate L1
    D = map(set, dataset)
    C1 = create_c1(dataset)
    L1, support_data = scan_d(dataset, C1, min_support)
    L = [L1]
    k = 2  # k means the next itemset
    # get Ck from previous L and filter Lk in loop
    while len(L[k - 2]) > 0:  # if current Li is [],stop the loop
        Ci = apriori_gen(L[k-2], k)
        Li, support_data_i = scan_d(D, Ci, min_support)
        L.append(Li)
        support_data.update(support_data_i)
        k += 1
    return L, support_data


def generate_rules(L, support_data, min_confidence=0.7):
    # traverse L and create H from L[i][j]
    big_rules = []
    for i in range(1, len(L)):
        for itemset in L[i]:
            H = [frozenset([item]) for item in itemset]
            # split 2_itemsets and n_itemsets(n>2)
            if i > 1:
                rules_from_conseq(itemset, H, support_data, big_rules, min_confidence)
            else:
                calc_conf(itemset, H, support_data, big_rules, min_confidence)
    return big_rules


def calc_conf(freq_set, H, support_data, brl, min_conf=0.7):
    # traverse H and calculate confidence of H-i --> i
    prunH = []  # association rules' consequent
    for item in H:
        conf = support_data[freq_set] / support_data[freq_set - item]
        if conf >= min_conf:
            print freq_set-item, "->", item, "confidence: ", conf
            brl.append((freq_set-item, item, conf))
            prunH.append(item)
    return prunH


def rules_from_conseq(freq_set, H, support_data, brl, min_conf=0.7):
    # get the len of first item of H
    m = len(H[0])
    if len(freq_set) > (m + 1):
        Hmp1 = apriori_gen(H, m+1)
        Hmp1 = calc_conf(freq_set, Hmp1, support_data, brl, min_conf)
        if len(Hmp1) > 1:
            rules_from_conseq(freq_set, Hmp1, support_data, brl, min_conf)
