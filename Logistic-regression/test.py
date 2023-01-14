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
    f,l = arrayGenReg()
    w_0 = np.array([[5.0,5.0,5.0]]).reshape(-1,1)
    w,w_res = w_cal(f,w_0,l,gd_cal=lr_gd,lr = 0.1,itera_times=50)
    print(w,w_res)
    loss_vis(f,w_res,l,MSELoss)

    w = sgd_cal(f,w_0,l,lr_gd,50)
    print(w)




    

main()
