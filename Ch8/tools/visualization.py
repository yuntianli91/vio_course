#coding=utf-8

import matplotlib.pyplot as plot
import numpy as np
import sys, getopt

N = len(sys.argv)
print("load %d files of solver time recorder." %(N-1))

fiilenames = sys.argv[1:]
x_labels = list()
# 读取文件
solver_time = list()
for file in fiilenames:
    tmp_time = np.loadtxt(file, dtype=np.float32)
    solver_time.append(tmp_time)
# 绘制求解器耗时统计图
#solver_time = np.loadtxt("../build/solver_cost.txt", dtype=np.float32)
fig_1 = plot.figure(figsize=(8,6))
ax = fig_1.add_subplot(111)

ax.violinplot(solver_time, showmeans=False, showmedians=True)
ax.yaxis.grid(True)
ax.set_xlabel("LM Methods")
ax.set_ylabel("cost(ms)")
# x轴画三个坐标
plot.setp(ax, xticks=[y + 1for y in range(len(solver_time))],
    xticklabels=fiilenames,);

plot.show();