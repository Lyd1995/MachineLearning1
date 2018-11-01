# Logistic 梯度下降算法练习
import numpy as np
import math

NUM_TRAIN = 1500
NUM_CHARACTER = 2
alpha = 0.01

def main():
    # X为输入训练集，有NUM_TRAIN个训练样本,每个样本有NUM_CHARATER个特征
    # Y是标签
    X = np.random.rand(NUM_TRAIN, NUM_CHARACTER)
    Y = [[int((x1 + x2) < 0.7)] for (x1, x2) in X]

    # 参数w和b
    w = np.random.rand(NUM_CHARACTER)
    b = 1

    # 开始使用梯度下降算法,一次迭代过程：
    dw = [0] * NUM_CHARACTER
    db = 0
    for i in range(NUM_TRAIN):
        z = np.dot(w, X[i]) + b  # z = wx + b
        a = 1/(1 + math.exp(-z))  # 计算预测值
        dz = a - Y[i][0]  # dz 表示 dJ/dz
        for j in range(NUM_CHARACTER):
            dw[j] = dw[j] + X[i][j] * dz
            db = db + dz

    db = db / NUM_TRAIN
    b = b - alpha * db
    for i in range(NUM_CHARACTER):
        dw[i] = dw[i] / NUM_TRAIN
        w[i] = w[i] - alpha * dw[i]

    # 一轮迭代过程结束
    print(w)
    print(b)

main()


