#coding=utf-8

import matplotlib.pyplot as plot
import numpy as np

# 绘制求解器耗时统计图
solver_time = np.loadtxt("../build/solver_cost.txt", dtype=np.float32)
fig_1 = plot.figure(figsize=(6,8))
ax = fig_1.add_subplot(111)

ax.violinplot(solver_time, showmeans=False, showmedians=True)
ax.yaxis.grid(True)
ax.set_xlabel("LM Methods")
ax.set_ylabel("cost(ms)")
plot.show();