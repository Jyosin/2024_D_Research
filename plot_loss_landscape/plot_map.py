import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

# 加载map数据
with open('/Users/wangruqin/Desktop/teacher_student/plot_loss_landscape/map_data.pkl', 'rb') as f:
    map = pickle.load(f)

# 准备绘图数据
x = np.array([item[0] for item in map])
y = np.array([item[1] for item in map])
z = np.array([item[2] for item in map])

# 创建网格
xi = np.linspace(min(x), max(x), 100)
yi = np.linspace(min(y), max(y), 100)
X, Y = np.meshgrid(xi, yi)

# 插值Z值
Z = griddata((x, y), z, (X, Y), method='cubic')

# 绘制3D曲面图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')

# 设置图表标题和轴标签
ax.set_title('Loss Landscape')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Loss')

# 添加颜色条
fig.colorbar(surf, shrink=0.5, aspect=5)

# 显示图表
plt.show()
