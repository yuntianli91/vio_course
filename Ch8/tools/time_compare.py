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
solver_time = list() # 每帧求解时间
hessian_time = list() # hessian时间
i=1
for file in fiilenames:
    tmp_time = np.loadtxt(file, dtype=np.float32)
    solver_time.append(tmp_time[...,0])
    hessian_time.append(tmp_time[...,1])
    # 获取文件名
    filepath, tmpfilename = os.path.split(sys.argv[i])
    shotname, extention = os.path.splitext(tmpfilename)
    x_labels.append(shotname)
    i = i + 1
# 使用pandas分析数据
df_solver = pd.DataFrame(solver_time)
df_hessian = pd.DataFrame(hessian_time)
print(df_solver.mean(1))
print(df_hessian.mean(1))
# 绘制求解器耗时统计图
#solver_time = np.loadtxt("../build/solver_cost.txt", dtype=np.float32)
fig, axes = plot.subplots(nrows = 2, ncols = 1, figsize=(10,8)) # 两行一列


axes[0].violinplot(solver_time, showmeans=False, showmedians=True)
axes[0].yaxis.grid(True)
axes[0].set_xlabel("solve frame")
axes[0].set_ylabel("cost(ms)")

axes[1].violinplot(hessian_time, showmeans=False, showmedians=True)
axes[1].yaxis.grid(True)
axes[1].set_xlabel("make hessian")
axes[1].set_ylabel("cost(ms)")
# x轴依次画标签三个坐标
plot.setp(axes, xticks=[y + 1for y in range(len(solver_time))],
    xticklabels=x_labels,);

plot.show();