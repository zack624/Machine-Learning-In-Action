# -*- coding: utf-8 -*-

import apriori
dataset = apriori.load_data_set()
c1 = apriori.create_c1(dataset)
d = map(set, dataset)
ret, support_data = apriori.scan_d(d, c1, 0.5)