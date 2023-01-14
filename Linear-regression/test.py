import numpy as np
import matplotlib.pyplot as plt

from Data_Generator import *
from Loss_Func import *

def main():
    #优化前
    f,l = arrayGenReg()
    plt.subplot(121)
    plt.scatter(f[:,0],l)
    plt.subplot(122)
    plt.scatter(f[:,1],l)
    #plt.show()
    w = np.array([[2,-1,-1]]).reshape(-1,1)
    sse = SSELoss(f,w,l)
    print(f.shape,w.shape,l.shape)
    print(sse)
    #优化后
    w_real = np.linalg.inv(f.T.dot(f)).dot(f.T).dot(l)
    sse_real = SSELoss(f,w_real,l)
    print(w_real)
    print(sse_real)
    plt.subplot(121)
    plt.scatter(f[:,0],f.dot(w_real))
    plt.subplot(122)
    plt.scatter(f[:,1],f.dot(w_real))
    plt.show()
    


main()
    
