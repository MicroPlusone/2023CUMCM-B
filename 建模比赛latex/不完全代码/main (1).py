import math

def calculate_coverage_width(D, theta, alpha):
    W1 = abs((D*math.tan(theta*math.pi/180))/(1-math.tan(theta*math.pi/180)*math.tan(alpha*math.pi/180)))
    W2 = abs((D * math.tan(theta*math.pi/180)) / (1 + math.tan(theta*math.pi/180) * math.tan(alpha*math.pi/180)))
    print(W1)
    print(W2)
    return W1+W2

def calculate_coverage_width2(D, theta, alpha):
    W2 = abs((D * math.tan(theta*math.pi/180)) / (1 + math.tan(theta*math.pi/180) * math.tan(alpha*math.pi/180)))
    return W2
def calculate_overlap_rate(W0,W1, distance,right):
    eta = (W0+right-distance)/W1
    return eta

def calculate_depth(h,alpha):
    a=70-h*math.tan(alpha*math.pi/180)
    return a

def main():
    theta = 120/2
    alpha = 1.5
    distance=200#测线间距步长
    h=800#距中心的距离
    W0 = 89.95#之前的右半个宽度
    a = calculate_depth(h, alpha)
    W1 = calculate_coverage_width(a, theta, alpha)
    d = W0 + W1 - distance  # 重叠部分面积
    right=calculate_coverage_width2(a, theta, alpha)
    eta = calculate_overlap_rate(W0,W1, distance,right)

    print(f"覆盖宽度 W: {W1:.2f} 米")
    print(f"深度 height： {a:.2f} 米")
    print(f"重叠率 eta: {eta * 100:.2f}%")

if __name__ == "__main__":
    main()
