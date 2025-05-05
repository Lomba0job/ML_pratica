
import numpy as np 

def data_gen():
    np.random.seed(1)
    X_xor = np.random.randn(200,2)
    y_xor = np.logical_xor(X_xor[:,0]>0, X_xor[:,1]>0)

    y_xor = np.where(y_xor, 1, -1)
    return X_xor, y_xor


import matplotlib.pyplot as plt

def data_show(X_xor, y_xor):

    plt.scatter(X_xor[y_xor == 1, 0], X_xor[y_xor == 1, 1], c='b', marker='x', label='1')
    plt.scatter(X_xor[y_xor == -1, 0], X_xor[y_xor == -1, 1], c='r', marker='o', label='-1')
    plt.xlim([-3,3])
    plt.ylim([-3,3])
    plt.legend(loc='best')
    plt.show()
    

