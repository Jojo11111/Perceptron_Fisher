import matplotlib.pyplot as plt
import numpy as np
from pylab import *


# 求类均值向量和类内离散度矩阵
def cal_cov_and_avg(samples):
    m = np.mean(samples, axis=0)
    m = np.mat(m)
    S = np.zeros((samples.shape[1], samples.shape[1]))
    S = np.mat(S)
    for j in samples:
        t = np.transpose(np.mat(j)) - np.transpose(m)
        S += t * np.transpose(t)
    return S, m


# Fisher判别准则
def fisher(x1, x2):
    S1, m1 = cal_cov_and_avg(x1)
    S2, m2 = cal_cov_and_avg(x2)
    Sw = S1 + S2
    Sw_inv = np.matrix(Sw).I
    return Sw_inv * np.transpose(m1 - m2)


# 判断样本类别 False代表第二类，True代表第一类
def judge(sample, w, x1, x2):
    m1 = np.mean(x1, axis=0)
    m1 = np.mat(m1)
    m1 = np.transpose(w) * np.transpose(m1)
    m2 = np.mean(x2, axis=0)
    m2 = np.mat(m2)
    m2 = np.transpose(w) * np.transpose(m2)
    w0 = -0.5 * (m1 + m2)
    if np.transpose(w) * np.transpose(sample) + w0 > 0:
        return True, w0
    else:
        return False, w0


x1 = np.array(
    [[1.24, 1.27], [1.36, 1.74], [1.38, 1.64], [1.38, 1.82], [1.38, 1.90], [1.40, 1.70], [1.48, 1.82], [1.54, 1.82],
     [1.56, 2.08]])
x2 = np.array([[1.14, 1.82], [1.18, 1.96], [1.20, 1.86], [1.26, 2.0], [1.28, 2.0], [1.3, 1.96]])

A = [1.24, 1.8]
B = [1.40, 2.04]
plt.scatter(A[0], A[1], c='g', marker='o')
plt.scatter(B[0], B[1], c='g', marker='o')
A = np.mat([1.24, 1.8])
B = np.mat([1.40, 2.04])
w = fisher(x1, x2)
outA, w0 = judge(np.array(A), w, x1, x2)
outB, w0 = judge(np.array(B), w, x1, x2)
print('A:',outA)
print('B:',outB)

plt.scatter(x1[:, 0], x1[:, 1], c='r', marker='o')
plt.scatter(x2[:, 0], x2[:, 1], c='b', marker='o')
x = np.arange(1.1, 1.6, 0.1)
y = double(w[0][0] / (-w[1][0])) * x + double(w0 / (-w[1][0]))
plt.plot(x, y, c='b')
plt.title('fisher')
plt.show()
