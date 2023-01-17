import numpy as np
import matplotlib.pyplot as plt

from Sigmoid import *
from Logic_Func import *
from Data_Generator import *

def main():
    """
    x = np.linspace(-10,10,100)
    y = sigmoid(x)
    plt.plot(x,y)
    plt.show()
    plt.plot(x,sigmoid_deri(x))
    plt.show()
    result = logic_cla(y)
    print(result)
    res = gd()
    show_trace(res)
    """
    """
    f,l = arrayGenReg()
    w_0 = np.array([[5.0,5.0,5.0]]).reshape(-1,1)
    w,w_res = w_cal(f,w_0,l,gd_cal=lr_gd,lr = 0.1,itera_times=50)
    print(w,w_res)
    loss_vis(f,w_res,l,MSELoss)

    w = sgd_cal(f,w_0,l,lr_gd,50)
    print(w)
    """
    f,l = arrayGenCla(300,2,2,[4,2],True)
    plt.scatter(f[:,0],f[:,1],c=l)
    plt.show()

    # 设置随机数种子
    np.random.seed(24)

    # 数据切分
    Xtrain, Xtest, ytrain, ytest = array_split(f, l)
    mean_ = Xtrain[:, :-1].mean(axis=0)
    std_ = Xtrain[:, :-1].std(axis=0)

    Xtrain[:, :-1] = (Xtrain[:, :-1] - mean_) / std_
    Xtest[:, :-1] = (Xtest[:, :-1] - mean_) / std_
    # 设置随机数种子
    np.random.seed(24)

    # 参数初始值
    n = f.shape[1]
    w = np.random.randn(n, 1)

    # 核心参数
    batch_size = 50
    num_epoch = 200
    lr_init = 0.2

    lr_lambda = lambda epoch: 0.95 ** epoch
    for i in range(num_epoch):
        w = sgd_cal(Xtrain, w, ytrain, logit_gd, batch_size=batch_size, epoch=1, lr=lr_init*lr_lambda(i))
    print(w)
    yhat = sigmoid(Xtrain.dot(w))
    re = logic_cla(yhat, thr=0.5)
    procent = (logic_cla(yhat, thr=0.5) == ytrain).mean()
    print(procent)

main()
