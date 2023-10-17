import math
#这个代码比第一问的代码好多了
beam_angle_deg = 120/2  # 多波束换能器的开角（单位：度）
slope_deg = 1.5  # 海底坡度（单位：度）
center_depth = 120  # 海域中心点处的海水深度（单位：米）
beta_deg = 90  # 测线方向夹角（单位：度）
distance_to_center = 0.9*1852  # 测量船距离海域中心点的距离（单位：米）
# 给定参数


# 将角度转换为弧度
beam_angle_rad = math.radians(beam_angle_deg)
slope_rad = math.radians(slope_deg)
beta_rad = math.radians(beta_deg)

deltaz=distance_to_center*math.cos(beta_rad)*math.tan(slope_rad)
depth=center_depth+deltaz
beta_rad=math.acos(math.cos(slope_rad)*math.cos(beta_rad))
print(depth)

# 计算波束覆盖宽度的数学模型
def calculate_beam_width(beam_angle_rad, slope_rad, beta_rad, depth):
    # 计算水深对应的波束覆盖宽度
    W1 = abs(
        (depth * math.tan(beam_angle_rad)) / (1 - math.tan(beam_angle_rad) * math.tan(slope_rad)))
    W2 = abs(
        (depth * math.tan(beam_angle_rad)) / (1 + math.tan(beam_angle_rad) * math.tan(slope_rad)))

    beam_width = W1+W2



    # 考虑坡度对波束宽度的影响
    beam_width_slope_adjusted = beam_width / math.cos(beta_rad)

    return beam_width_slope_adjusted

a=calculate_beam_width(beam_angle_rad, slope_rad,beta_rad, depth)
# 计算多波束测深的覆盖宽度
coverage_width = calculate_beam_width(beam_angle_rad, slope_rad, beta_rad,depth)

# 计算相邻测线之间的间距
overlap_percentage = 15  # 相邻条带之间的重叠率（15%）

# 计算相邻测线的间距（d）
d = coverage_width * (100 / overlap_percentage - 1)

# 打印结果
print(f"多波束测深的覆盖宽度：{a:.2f} 米")

