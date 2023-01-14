import numpy as np

def arrayGenReg(num_examples = 1000, w = [2, -1, 1], bias = True, delta = 3, deg = 1):
    """回归类数据集创建函数。

    :param num_examples: 创建数据集的数据量
    :param w: 包括截距的（如果存在）特征系数向量
    :param bias：是否需要截距
    :param delta：扰动项取值
    :param deg：方程最高项次数
    :return: 生成的特征数组和标签数组
    """
    
    if bias == True:
        num_inputs = len(w)-1                                                           # 数据集特征个数
        features_true = np.random.randn(num_examples, num_inputs)                       # 原始特征
        w_true = np.array(w[:-1]).reshape(-1, 1)                                        # 自变量系数
        b_true = np.array(w[-1])                                                        # 截距
        labels_true = np.power(features_true, deg).dot(w_true) + b_true                 # 严格满足人造规律的标签
        features = np.concatenate((features_true, np.ones_like(labels_true)), axis=1)    # 加上全为1的一列之后的特征
    else: 
        num_inputs = len(w)
        features = np.random.randn(num_examples, num_inputs) 
        w_true = np.array(w).reshape(-1, 1)         
        labels_true = np.power(features, deg).dot(w_true)
    labels = labels_true + np.random.normal(size = labels_true.shape) * delta
    return features, labels
