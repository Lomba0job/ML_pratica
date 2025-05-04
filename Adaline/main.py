
import os 

import pandas as pd 
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


from neurone import ADalineGD

df = pd.read_csv('iris.data', header=None, encoding='utf-8')
df.tail()

y = df.iloc[0:100, 4].values
y  = np.where(y == 'Iris-setosa', -1, 1)

X = df.iloc[0:100, [0, 2]].values

"""
STANDARDIZZAZIONE figura 2

X_std = np.copy(X)
X_std[:,0] = (X[:,0] - X[:,0].mean()) /  X[:,0].std()
X_std[:,1] = (X[:,1] - X[:,1].mean()) /   X[:,1].std() 


fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)


ada1 = ADalineGD(eta=0.0001, n_iter=15)
ada1.fit(X, y)
ax1.plot(range(1, len(ada1.cost_) + 1 ), ada1.cost_, marker='o')
ax1.set_title("Adaline - learning rate 0.0001- X")

ada2 = ADalineGD(eta=0.01, n_iter=15)
ada2.fit(X, y)
ax2.plot(range(1, len(ada2.cost_) + 1 ), ada2.cost_, marker='o')
ax2.set_title("Adaline - learning rate 0.01 - X")

ada3 = ADalineGD(eta=0.0001, n_iter=15)
ada3.fit(X_std, y)
ax3.plot(range(1, len(ada3.cost_) + 1 ), ada3.cost_, marker='o')
ax3.set_title("Adaline - learning rate 0.0001 - X_std")

ada4 = ADalineGD(eta=0.01, n_iter=15)
ada4.fit(X_std, y)
ax4.plot(range(1, len(ada4.cost_) + 1 ), ada4.cost_, marker='o')
ax4.set_title("Adaline - learning rate 0.01 - X_std")


"""



fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('DATI -> PERCE')

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
X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()




ada_gd = ADalineGD(n_iter=15, eta=0.01)
ada_gd.fit(X_std, y)

plot_decision_region(X_std, y, classifier=ada_gd)
plt.title('Adaline - Gradient Descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
# plt.savefig('images/02_14_1.png', dpi=300)
plt.show()

plt.plot(range(1, len(ada_gd.cost_) + 1), ada_gd.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Sum-squared-error')

plt.tight_layout()
# plt.savefig('images/02_14_2.png', dpi=300)
plt.show()