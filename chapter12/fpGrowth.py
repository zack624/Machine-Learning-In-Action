# -*- coding: utf-8 -*-


class TreeNode:
    def __init__(self, name_value, num_occur, parent_node):
        self.name = name_value
        self.count = num_occur
        self.node_link = None
        self.parent = parent_node
        self.children = {}

    def inc(self, num):
        self.count += num

    def disp(self, i=1):
        print " " * i, self.name, " ", self.count
        for child in self.children.values():
            child.disp(i + 1)


def load_simple_data():
    return [['r', 'z', 'h', 'j', 'p'],
        ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
        ['z'],
        ['r', 'x', 'n', 'o', 's'],
        ['y', 'r', 'x', 'z', 'q', 't', 'p'],
        ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]


def create_init_set(dataset):
    ret_dict = {}
    for tran in dataset:
        ret_dict[frozenset(tran)] = 1
    return ret_dict


def create_tree(dataset, min_support=1):
    # generate header table
    header_table = {}
    for tran in dataset:
        for item in tran:
            header_table[item] = header_table.get(item, 0) + dataset[tran]
    for header in header_table:
        if header_table[header] < min_support:
            del(header_table[header])
    freq_items = set(header_table.keys())
    if len(freq_items) == 0:
        return None, None
    for header in header_table:
        header_table[header] = [header_table[header], None]
    # sort header table and update tree
    ret_tree = TreeNode("Null Set", 1, None)
    for tran, count in dataset.items():
        local_dict = {}
        for item in tran:
            if item in freq_items:
                local_dict[item] = header_table[item]
        if len(local_dict) > 0:
            sorted_items = [item[0] for item in sorted(local_dict.items(), lambda p:p[1], reverse=True)]
            update_tree(sorted_items, ret_tree, header_table, count)
    return ret_tree, header_table


def update_tree(sorted_items, tree, header_table, count):
    pass





