
import os 

import pandas as pd 
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


from neurone import Perceptron

df = pd.read_csv('iris.data', header=None, encoding='utf-8')
df.tail()

y = df.iloc[0:100, 4].values
y  = np.where(y == 'Iris-setosa', -1, 1)

X = df.iloc[0:100, [0, 2]].values

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('DATI -> PERCE')



# dati
ax1.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
ax1.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')

ax1.set_xlabel('lunghezza gambo [cm]')
ax1.set_ylabel('lunghezza petali [cm]')

ax1.legend(loc='upper left')



# auto apprendimento
ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)
#
#plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
#
#plt.xlabel('Epoche')
#plt.ylabel('Numero di aggiornamenti')
#plt.show()

def plot_decision_region(X, y, classifier, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    x1_min, x1_max = X[:, 0].min() -1,  X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() -1,  X[:, 1].max() + 1
    
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    ax2.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    ax2.set_xlim(xx1.min(), xx1.max())
    ax2.set_ylim(xx2.min(), xx2.max())
    ax1.set_xlim(xx1.min(), xx1.max())
    ax1.set_ylim(xx2.min(), xx2.max())
    
    for idx, cl in enumerate(np.unique(y)):
        ax2.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1], 
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl, 
                    edgecolors='black')


plot_decision_region(X, y, classifier=ppn)
ax2.set_xlabel('lunghezza gambo [cm]')
ax2.set_ylabel('lunghezza petali [cm]')

ax2.legend(loc='upper left')
plt.show()