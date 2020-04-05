#coding=utf-8

import matplotlib.pyplot as plot
import numpy as np
import pandas as pd
import sys, getopt, os

N = len(sys.argv)
print("load %d files of solver time recorder." %(N-1))

fiilenames = sys.argv[1:]
x_labels = list()



# 读取文件
solver_time = list()
i=1
for file in fiilenames:
    tmp_time = np.loadtxt(file, dtype=np.float32)
    solver_time.append(tmp_time)
    # 获取文件名
    filepath, tmpfilename = os.path.split(sys.argv[i])
    shotname, extention = os.path.splitext(tmpfilename)
    x_labels.append(shotname)
    i = i + 1
# 使用pandas分析数据
df = pd.DataFrame(solver_time)
print(df.mean(1))
# 绘制求解器耗时统计图
#solver_time = np.loadtxt("../build/solver_cost.txt", dtype=np.float32)
fig_1 = plot.figure(figsize=(10,8))
ax = fig_1.add_subplot(111)

ax.violinplot(solver_time, showmeans=False, showmedians=True)
ax.yaxis.grid(True)
ax.set_xlabel("LM Methods")
ax.set_ylabel("cost(ms)")
# x轴画三个坐标
plot.setp(ax, xticks=[y + 1for y in range(len(solver_time))],
    xticklabels=x_labels,);

plot.show();