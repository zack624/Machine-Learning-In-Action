# -*- coding: utf-8 -*-

import fpGrowth
tree = fpGrowth.TreeNode("pyramid", 9, None)
tree.children['eye'] = fpGrowth.TreeNode("eye", 13, None)
tree.children['phoenix'] = fpGrowth.TreeNode("phoenix", 3, None)
tree.disp()

import fpGrowth
ds = fpGrowth.create_init_set(fpGrowth.load_simple_data())