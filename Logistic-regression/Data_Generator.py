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

def arrayGenCla(num_examples = 500, num_inputs = 2, num_class = 3, deg_dispersion = [4, 2], bias = False):
    """分类数据集创建函数。
    
    :param num_examples: 每个类别的数据数量
    :param num_inputs: 数据集特征数量
    :param num_class：数据集标签类别总数
    :param deg_dispersion：数据分布离散程度参数，需要输入一个列表，其中第一个参数表示每个类别数组均值的参考、第二个参数表示随机数组标准差。
    :param bias：建立模型逻辑回归模型时是否带入截距，为True时将添加一列取值全为1的列
    :return: 生成的特征张量和标签张量，其中特征张量是浮点型二维数组，标签张量是长正型二维数组。
    """
    
    cluster_l = np.empty([num_examples, 1])                            # 每一类标签数组的形状
    mean_ = deg_dispersion[0]                                        # 每一类特征数组的均值的参考值
    std_ = deg_dispersion[1]                                         # 每一类特征数组的方差
    lf = []                                                          # 用于存储每一类特征的列表容器
    ll = []                                                          # 用于存储每一类标签的列表容器
    k = mean_ * (num_class-1) / 2                                    # 每一类特征均值的惩罚因子
    
    for i in range(num_class):
        data_temp = np.random.normal(i*mean_-k, std_, size=(num_examples, num_inputs))     # 生成每一类特征
        lf.append(data_temp)                                                               # 将每一类特征添加到lf中
        labels_temp = np.full_like(cluster_l, i)                                           # 生成某一类的标签
        ll.append(labels_temp)                                                             # 将每一类标签添加到ll中
        
    features = np.concatenate(lf,1).reshape(num_examples,num_class,num_inputs)
    labels = np.concatenate(ll,1).reshape(num_examples,num_class,1)
    
    if bias == True:
        features = np.concatenate((features, np.ones(labels.shape)), 2)   # 在特征张量中添加一列全是1的列
    return features, labels



