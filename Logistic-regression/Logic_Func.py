import numpy as np
import matplotlib.pyplot as plt
from Loss_Func import *

def logic_cla(yhat, thr=0.5):
    """
    逻辑回归类别输出函数：
    :param yhat: 模型输出结果
    :param thr：阈值
    :return ycla：类别判别结果
    """
    ycla = np.zeros_like(yhat)
    ycla[yhat >= thr] = 1
    return ycla

def gd(lr = 0.02, itera_times = 20, w = 10):
    """
    梯度下降计算函数
    :param lr: 学习率
    :param itera_times：迭代次数
    :param w：参数初始取值
    :return results：每一轮迭代的参数计算结果列表
    """                              
    results = [w]
    for i in range(itera_times):
        w -= lr * 28 * (w - 2)            # 梯度计算公式
        results.append(w)
    return results

def show_trace(res):
    """
    梯度下降轨迹绘制函数
    """
    f_line = np.arange(-6, 10, 0.1)
    plt.plot(f_line, [14 * np.power(x-2, 2) for x in f_line])
    plt.plot(res, [14 * np.power(x-2, 2) for x in res], '-o')
    plt.xlabel('x')
    plt.ylabel('Loss(x)')
    plt.show()

def MSELoss(X, w, y):
    """
    MSE指标计算函数
    """
    SSE = SSELoss(X, w, y)
    MSE = SSE / X.shape[0]
    return MSE

def lr_gd(X, w, y):
    """
    线性回归梯度计算公式
    """
    m = X.shape[0]
    grad = 2 * X.T.dot((X.dot(w) - y)) / m
    return grad

def w_cal(X, w, y, gd_cal, lr = 0.02, itera_times = 20):
    """   
    梯度下降中参数更新函数 
    :param X: 训练数据特征
    :param w: 初始参数取值
    :param y: 训练数据标签
    :param gd_cal：梯度计算公式
    :param lr: 学习率
    :param itera_times: 迭代次数       
    :return w：最终参数计算结果   
    """
    w_res = [np.copy(w)]
    for i in range(itera_times):
        w -= lr * gd_cal(X, w, y)
        w_res.append(np.copy(w))
    return w,w_res

def loss_vis(X, w_res, y, loss_func):
    loss_value = np.array([loss_func(X, np.array(w), y) for w in w_res]).flatten()
    plt.plot(np.arange(len(loss_value)), loss_value)
    plt.show()

def sgd_cal(X, w, y, gd_cal, epoch, batch_size=1, lr=0.02, shuffle=True, random_state=24):
    """
    随机梯度下降和小批量梯度下降计算函数
    :param X: 训练数据特征
    :param w: 初始参数取值
    :param y: 训练数据标签
    :param gd_cal：梯度计算公式
    :param epoch: 遍历数据集次数
    :batch_size: 每一个小批包含数据集的数量
    :param lr: 学习率
    :shuffle：是否在每个epoch开始前对数据集进行乱序处理
    :random_state：随机数种子值
    :return w：最终参数计算结果       
    """
    m = X.shape[0]
    n = X.shape[1]
    batch_num = np.ceil(m / batch_size)
    X = np.copy(X)
    y = np.copy(y)
    for j in range(epoch):
        if shuffle:            
            np.random.seed(random_state)                           
            np.random.shuffle(X)                            
            np.random.seed(random_state)
            np.random.shuffle(y)    
        for i in range(np.int(batch_num)):
            w = w_cal(X[i*batch_size: np.min([(i+1)*batch_size, m])], 
                      w, 
                      y[i*batch_size: np.min([(i+1)*batch_size, m])], 
                      gd_cal=gd_cal, 
                      lr=lr, 
                      itera_times=1)
    return w
