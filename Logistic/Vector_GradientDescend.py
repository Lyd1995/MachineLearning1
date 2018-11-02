# Logistic 梯度下降算法练习----向量化
import numpy as np
import math

NUM_TRAIN = 1500
NUM_CHARACTER = 2
alpha = 0.01

def main():
    SEED = 23455
    # 基于seed产生随机数
    rdm = np.random.RandomState(SEED)
    # X为输入训练集，有NUM_TRAIN个训练样本,每个样本有NUM_CHARATER个特征
    # Y是标签
    X = rdm.rand(NUM_CHARACTER, NUM_TRAIN)
    Y = np.zeros(shape=[1, NUM_TRAIN])
    # Y = np.zeros(shape=[1, NUM_TRAIN], dtype=np.int)   如果指定数据类型，计算出来的结果会与非向量化的结算结果不同，这因为数据类型不一样导致的

    # print(X.T.shape)
    i = 0
    for (x1, x2) in X.T:
        if x1 + x2 < 0.6:
            Y[0][i] = 1
        i += 1

    # 参数w和b
    W = rdm.rand(NUM_CHARACTER, 1)
    b = 1

    # 开始使用梯度下降算法,一次迭代过程：
    Z = np.dot(W.T, X) + b
    A = 1/(1 + np.exp(-Z))

    # print(A.shape)
    dZ = A - Y
    dW = np.dot(X, dZ.T) * 1 / NUM_TRAIN
    db = dZ.sum() * 1 / NUM_TRAIN

    b = b - alpha * db
    W = W - alpha * dW

    # 一轮迭代过程结束
    print(W)
    print(b)

main()
