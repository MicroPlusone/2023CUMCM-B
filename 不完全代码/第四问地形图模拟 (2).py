from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

data = pd.read_excel(r"C:/Users/MMinuzero/Desktop/B题/附件.xlsx")

x=data.iloc[0:1, 2:203].values
y=data.iloc[1:253, 1:2].values
X = np.array(x)
Y = np.array(y)
X, Y = np.meshgrid(X, Y)
print("网格化后的X=",X)
print("X维度信息",X.shape)
print("网格化后的Y=",Y)
print("Y维度信息", Y.shape)
z=data.iloc[1:253, 2:203].values
Z = np.array(z)
print("网格化后的Z=",Z)
print("Z轴数据维度",Z.shape)
sns.heatmap(-Z)

plt.show()
fig=plt.figure()
ax = fig.add_subplot(projection = '3d')
surf=ax.plot_surface(X,Y,Z,cmap='BuPu',linewidth=0,antialiased=False)
fig.colorbar(surf)
ax.set_xlabel('N_S(n.m.)', color='b')
ax.set_ylabel('W_E(n.m.)', color='g')
ax.set_zlabel('Depth(m)', color='r')
ax.set_title('海水深度三维图')
plt.show()

print("完成")

