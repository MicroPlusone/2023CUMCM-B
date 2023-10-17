from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from scipy import interpolate

###########导入数据，初始化，网格化#########################
data = pd.read_excel(r"E:\大学\数学模型\2023国赛\国赛提交\过程文件\附件 - 副本.xlsx")

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

##########依据等高线选代表梯度的点#########################
# 找到最大的z值及其对应的坐标
max_z = np.max(Z)
max_indices = np.argwhere(Z == max_z)
x_0, y_0 = X[max_indices[0][0], max_indices[0][1]], Y[max_indices[0][0], max_indices[0][1]]

# 打印z值最大的点的坐标，并标注为A点
print(f"A点坐标：({x_0}, {y_0}, {max_z})")

# 设置步长d和弹性范围c
d = 10
c = 0.3

# 初始化结果列表
selected_points = []

# 开始搜索最近的点
z_i = max_z
while True:
    z_i -= d
    z_min = z_i - c
    z_max = z_i + c
    
    # 找到z范围内的点
    within_range_indices = np.argwhere((Z >= z_min) & (Z <= z_max))
    
    if len(within_range_indices) == 0:
        break  # 如果没有点在范围内，退出循环
    
    # 计算距离最近的点
    distances = np.sqrt((X[within_range_indices[:, 0], within_range_indices[:, 1]] - x_0)**2 +
                        (Y[within_range_indices[:, 0], within_range_indices[:, 1]] - y_0)**2 +
                        (Z[within_range_indices[:, 0], within_range_indices[:, 1]] - z_i)**2)
    min_distance_index = np.argmin(distances)
    
    # 获取最近点的坐标
    x_i, y_i, z_i = X[within_range_indices[min_distance_index][0],within_range_indices[min_distance_index][1]],Y[within_range_indices[min_distance_index][0],within_range_indices[min_distance_index][1]], Z[within_range_indices[min_distance_index][0], within_range_indices[min_distance_index][1]]
    
    # 添加到结果列表
    selected_points.append((x_i, y_i, z_i))

# 重新编号所选点
for i, (x_i, y_i, z_i) in enumerate(selected_points):
    print(f"A_{i+1}坐标：({x_i}, {y_i}, {z_i})")

#########################可视化选点结果############################

# 提取selected_points中的z值并排序
z_values = sorted(list(set([point[2] for point in selected_points])))

# 在等高线图上绘制等高线，使用排序后的z值
plt.contourf(X, Y, Z, levels=z_values + [z_values[-1] + 1], cmap='viridis')
plt.colorbar()

# 绘制选定的点及其投影坐标
for i, (x_i, y_i, z_i) in enumerate(selected_points):
    # 绘制选定点的标记
    plt.scatter(x_i, y_i, c='red', label=f'A_{i+1}', s=50, zorder=3)

# 设置y轴数据范围为（0,5）
plt.ylim(0, 5)

# 设置图形属性
plt.xlabel('X')
plt.ylabel('Y')
plt.title('contour map')
plt.legend()
plt.grid(True)
plt.show()

##################做拟合，找梯度方向########################
# 提取A点的x和y坐标
x_coords = [point[0] for point in selected_points]
y_coords = [point[1] for point in selected_points]

# 进行线性回归
coefficients = np.polyfit(x_coords, y_coords, 1)
slope, intercept = coefficients

# 创建拟合直线的x值范围
x_fit = np.linspace(min(x_coords), max(x_coords), 100)
y_fit = slope * x_fit + intercept

# 绘制等高线图
plt.contourf(X, Y, Z, levels=z_values + [z_values[-1] + 1], cmap='viridis')
plt.colorbar()

# 绘制选定的点及其投影坐标
for i, (x_i, y_i, z_i) in enumerate(selected_points):
    # 绘制选定点的标记
    plt.scatter(x_i, y_i, c='red', label=f'A_{i+1}', s=50, zorder=3)
    
# 绘制线性回归的拟合直线
plt.plot(x_fit, y_fit, 'b-', label=f'Linear Fit (y = {slope:.2f}x + {intercept:.2f})', lw=2)

# 设置y轴数据范围为（0,5）
plt.ylim(0, 5)
#记录拟合直线数据
y = -0.15 * x + 0.69

# 设置图形属性
plt.xlabel('X轴')
plt.ylabel('Y轴')
plt.title('Linear fitting')
plt.legend()
plt.grid(True)
plt.show()

##########################找B_j#############################
# 初始化结果列表
selected_B_j_points = []

# 遍历 X、Y、Z 数据
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        x_i = X[i, j]
        y_i = Y[i, j]
        z_i = Z[i, j]
        
        # 计算每个点的 y_j 值
        calculated_y_j = -0.15 * x_i + 0.69
        
        # 检查是否满足条件
        if abs(calculated_y_j - y_i) < 1e-6:  # 使用小误差来比较浮点数
            selected_B_j_points.append((x_i, y_i, z_i))

# 按照 x_j 从小到大的顺序对点进行排序
sorted_B_j_points = sorted(selected_B_j_points, key=lambda point: point[0])

# 打印排序后的 B_j 点
for i, (x_j, y_j, z_j) in enumerate(sorted_B_j_points):
    print(f"B_{i+1}坐标：({x_j}, {y_j}, {z_j})")

##############定义第j条测线在xoy平面投影的表达式###############
#y = (1/0.15) * (x - x_j) + y_j  ####斜率6.66，网格化B_j两侧做3*2网格

##############定义梯度拟合线上点B_j处的坡度###############
# 初始化结果列表
selected_B_j_points = []

# 遍历 X、Y、Z 数据
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        x_i = X[i, j]
        y_i = Y[i, j]
        z_i = Z[i, j]
        
        # 计算每个点的 y_j 值
        calculated_y_j = -0.15 * x_i + 0.69
        
        # 检查是否满足条件
        if abs(calculated_y_j - y_i) < 1e-6:  # 使用小误差来比较浮点数
            selected_B_j_points.append((x_i, y_i, z_i))

# 按照 x_j 从小到大的顺序对点进行排序
sorted_B_j_points = sorted(selected_B_j_points, key=lambda point: point[0])

# 打印排序后的 B_j 点
for i, (x_j, y_j, z_j) in enumerate(sorted_B_j_points):
    print(f"B_{i+1}坐标：({x_j}, {y_j}, {z_j})")
########################插值############################
# 将 B_j 点的坐标拆分成 x、y、z 列表
x_j_values = [point[0] for point in sorted_B_j_points]
y_j_values = [point[1] for point in sorted_B_j_points]
z_j_values = [point[2] for point in sorted_B_j_points]

# 创建一个列表，用于存储插值后的点
interpolated_points = []

# 在相邻的 B_j 点之间进行插值
for i in range(len(x_j_values) - 1):
    num_interpolated_points = 20  # 插入的点数
    x_interp = np.linspace(x_j_values[i], x_j_values[i + 1], num_interpolated_points)
    y_interp = np.linspace(y_j_values[i], y_j_values[i + 1], num_interpolated_points)
    z_interp = np.linspace(z_j_values[i], z_j_values[i + 1], num_interpolated_points)
    
    # 将插值结果添加到列表中
    interpolated_points.extend(list(zip(x_interp, y_interp, z_interp)))

# 合并原始 B_j 点和插值后的点
all_points = sorted_B_j_points + interpolated_points

# 使用 sorted 函数按照 x 值从大到小排序
sorted_all_points = sorted(all_points, key=lambda point: point[0], reverse=True)

# 为排列后的点标记为C_m
C_m_points = [(f'C_{i}', x_i, y_i, z_i) for i, (x_i, y_i, z_i) in enumerate(sorted_all_points, start=1)]

# 创建一个列表，用于存储 C_m 的信息
C_m = []

# 输出标记后的点
for label, x_i, y_i, z_i in C_m_points:
    # 将坐标信息添加到 C_m 列表中
    C_m.append({'x_m': x_i, 'y_m': y_i, 'z_m': z_i})

# 输出标记后的点
for label, x_i, y_i, z_i in C_m_points:
    print(f"{label}坐标：({x_i}, {y_i}, {z_i})")
################定义坡度##################
# 计算每个 C_m 点的坡度
slope_values = []
for i in range(5, len(C_m_points) - 5):
    x_m = C_m_points[i][1]
    y_m = C_m_points[i][2]
    z_m_minus_5 = C_m_points[i - 5][3]
    z_m_plus_5 = C_m_points[i + 5][3]
    distance_squared = (C_m_points[i + 5][1] - C_m_points[i - 5][1])**2 + (C_m_points[i + 5][2] - C_m_points[i - 5][2])**2
    tana_m = (z_m_minus_5 - z_m_plus_5) / (1852 * np.sqrt(distance_squared))
    slope_values.append(tana_m)

# 打印每个 C_m 点的坡度值
for i, slope in enumerate(slope_values, start=5):
    print(f"C_{i}点的坡度 (tana_m)：{slope:.4f}")

################定义C_m处的覆盖宽度########################
# 存储覆盖宽度1（W1_m）和覆盖宽度2（W2_m）的值
W1_values = []
W2_values = []
# 存储 C_m 点的坐标和宽度信息
C_m = []

# 计算每个 C_m 点的 W1_m 和 W2_m
for i in range(5, len(C_m_points) - 5):
    z_m = C_m_points[i][3]
    tana_m = slope_values[i - 5]
    
    # 计算覆盖宽度1（W1_m）
    W1_m = (z_m * np.sqrt(3)) / (1 - 3 * tana_m)/1852
    
    # 计算覆盖宽度2（W2_m）
    W2_m = (z_m * np.sqrt(3)) / (1 + 3 * tana_m)/1852
    
    # 将坐标和宽度信息添加到 C_m 列表中
    C_m.append({'x_m': C_m_points[i][1], 'y_m': C_m_points[i][2], 'z_m': z_m, 'W1_m': W1_m, 'W2_m': W2_m})

# 打印每个 C_m 点的坐标、W1_m 和 W2_m 值
for i, point in enumerate(C_m, start=5):
    print(f"C_{i}坐标：(x_m={point['x_m']}, y_m={point['y_m']}, z_m={point['z_m']}), W1_m={point['W1_m']:.4f}, W2_m={point['W2_m']:.4f}")
    # 存储到相应的列表中
    W1_values.append(W1_m)
    W2_values.append(W2_m)

# 打印每个 C_m 点的 W1_m 和 W2_m 值
for i, (W1, W2) in enumerate(zip(W1_values, W2_values), start=5):
    print(f"C_{i}点的覆盖宽度1 (W1_m)：{W1:.4f}")
    print(f"C_{i}点的覆盖宽度2 (W2_m)：{W2:.4f}")

###############选择测线##########################
# 初始化选定的C_m点
selected_C_m = None
max_distance = 0  # 用于跟踪最远距离

# 假设D0点的坐标为 (x_1, y_1)
x_1 = C_m_points[0][1]
y_1 = C_m_points[0][2]

# 计算D0点的覆盖宽度1（W1_1）和覆盖宽度2（W2_1）
W1_1 = W1_values[0] if len(W1_values) > 0 else 0
W2_1 = W2_values[0] if len(W2_values) > 0 else 0

# 遍历C_m点，查找满足条件的点
for i in range(len(C_m_points)):
    x_m1 = C_m_points[i][1]
    y_m1 = C_m_points[i][2]
    
    # 确保索引在有效范围内
    if i < len(W1_values) and i < len(W2_values):
        W1_m1 = W1_values[i]
        W2_m1 = W2_values[i]
        
        # 计算条件的值
        condition_value = (W2_1 + W1_m - np.sqrt((x_1 - x_m1)**2 + (y_1 - y_m1)**2)) / (W1_m + W2_m)
        
        # 检查是否满足条件
        if 0.13 <= condition_value <= 0.17:
            # 计算距离D0点的距离
            distance_to_D0 = np.sqrt((x_1 - x_m1)**2 + (y_1 - y_m1)**2)
            
            # 如果距离更远，更新选定的C_m点和最远距离
            if distance_to_D0 > max_distance:
                selected_C_m = C_m_points[i]
                max_distance = distance_to_D0

if selected_C_m is not None:
    print(f"选择的C_m点坐标：({selected_C_m[1]}, {selected_C_m[2]}, {selected_C_m[3]})")
else:
    print("未找到满足条件的C_m点。")  

