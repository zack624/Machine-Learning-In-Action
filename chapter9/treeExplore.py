# -*- coding: utf-8 -*-

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from numpy import *
from Tkinter import *
import regTrees


def re_draw(tols, toln):
    re_draw.f.clf()
    re_draw.a = re_draw.f.add_subplot(111)
    if chk_button_var.get():
        if toln < 2: toln = 2
        my_tree = regTrees.create_tree(re_draw.raw_data, regTrees.model_leaf, regTrees.model_err, (tols, toln))
        yhat = regTrees.create_forecast(my_tree, re_draw.test_data, regTrees.model_tree_eval)
    else:
        my_tree = regTrees.create_tree(re_draw.raw_data, ops=(tols, toln))
        yhat = regTrees.create_forecast(my_tree, re_draw.test_data)
    re_draw.a.scatter(re_draw.raw_data[:, 0].tolist(), re_draw.raw_data[:, 1].tolist(), s=5)
    re_draw.a.plot(re_draw.test_data, yhat, linewidth=2.0)
    re_draw.canvas.show()


def get_inputs():
    try: toln = int(n_entry.get())
    except:
        toln = 10
        print "enter Integer for tolN"
        n_entry.delete(0, END)
        n_entry.insert(0, '10')
    try: tols = float(s_entry.get())
    except:
        tols = 1.0
        print "enter Float for tolS"
        s_entry.delete(0, END)
        s_entry.insert(0, '1.0')
    return toln, tols


def draw_new_tree():
    toln, tols = get_inputs()
    re_draw(tols, toln)


root = Tk()
Label(root, text="Plot Place Holder").grid(row=0, columnspan=3)
Label(root, text="tolN").grid(row=1, column=0)
n_entry = Entry(root)
n_entry.grid(row=1, column=1)
n_entry.insert(0, '10')
Label(root, text='tolS').grid(row=2, column=0)
s_entry = Entry(root)
s_entry.grid(row=2, column=1)
s_entry.insert(0, '1.0')
Button(root, text="ReDraw", command=draw_new_tree).grid(row=1, column=2, columnspan=3)

chk_button_var = IntVar()
chk_button = Checkbutton(root, text="Model Tree", variable=chk_button_var)
chk_button.grid(row=3, column=0, columnspan=2)

re_draw.raw_data = mat(regTrees.load_data_set('sine.txt'))
re_draw.test_data = arange(min(re_draw.raw_data[:, 0]), max(re_draw.raw_data[:, 0]), 0.01)
# arange等同于range,返回的是array数组
re_draw.f = Figure(figsize=(5, 4), dpi=100)
re_draw.canvas = FigureCanvasTkAgg(re_draw.f, master=root)
re_draw.canvas.show()
re_draw.canvas.get_tk_widget().grid(row=0, columnspan=3)
re_draw(1.0, 10)

root.mainloop()