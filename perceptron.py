from pylab import *

x1 = [[1.24, 1.27], [1.36, 1.74], [1.38, 1.64], [1.38, 1.82], [1.38, 1.90], [1.40, 1.70], [1.48, 1.82], [1.54, 1.82],
      [1.56, 2.08]]
x2 = [[1.14, 1.82], [1.18, 1.96], [1.20, 1.86], [1.26, 2.0], [1.28, 2.0], [1.3, 1.96]]

# 规范化增广样本向量
for i in range(len(x1)):
    plt.scatter(x1[i][0], x1[i][1], c='r', marker='*')
    x1[i].append(1)
    x1[i] = np.transpose(np.mat(x1[i]))
for j in range(len(x2)):
    plt.scatter(x2[j][0], x2[j][1], c='b', marker='^')
    x2[j].append(1)
    x2[j][0] *= -1
    x2[j][1] *= -1
    x2[j][2] *= -1
    x2[j] = np.transpose(np.mat(x2[j]))
Y = x1 + x2

# 迭代求a
l = len(Y)
a = np.transpose(np.mat([0, 0, 0]))
n = 0
flag = 0
while flag < l:
    n = 0
    flag = 0
    while n < l:
        if np.transpose(a) * Y[n] <= 0:
            a = a + Y[n]
            print('a=', a)
            n += 1
        else:
            flag += 1
            n += 1

# 样本类别判断
A = [1.24, 1.8, 1]
B = [1.40, 2.04, 1]
plt.scatter(A[0], A[1], c='g', marker='o')
plt.scatter(B[0], B[1], c='g', marker='o')
A = np.transpose(np.mat(A))
B = np.transpose(np.mat(B))
if np.transpose(a) * A > 0:
    print("A belongs to x1")
else:
    print("A belongs to x2")
if np.transpose(a) * B > 0:
    print("B belongs to x1")
else:
    print("B belongs to x2")

# 画图
x = np.arange(1.1, 1.7, 0.1)
y = double(a[0][0] / (-a[1][0])) * x + double(1 / (-a[1][0]))
plt.plot(x, y, c='k')
plt.title('perceptron')
plt.show()
