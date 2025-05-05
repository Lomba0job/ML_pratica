from sklearn import datasets
import numpy as np

iris = datasets.load_iris() #iris dataset molto comune quindi già implementato all' intenro della libreria 
X = iris.data[:, [2, 3]]
y = iris.target
print("Class labels: ", np.unique(y)) # Class Labels [0 1 2] 
""" 
Pratica comune quella di classificare le etichette con numeri interi, 
poichè più facile da gestire e più veloce da calcolare.
"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

print ("Labels conunts in y: ", np.bincount(y)) #Lavels counts in y:  [50 50 50]
print ("Labels counts in y_train: ", np.bincount(y_train)) # Labels counts in y_train:  [35 35 35]
print ("Labels counts in y_test: ", np.bincount(y_test)) # Labels counts in y_test:  [15 15 15]

""" 
suddividermo casualmente il nostro dataset in due parti, una per l' addestramento e una per il test.
In questo caso il 70% dei dati sarà usato per l' addestramento e il 30% per il test.
Il parametro random_state serve per fissare un sempre casuale fissom, in modo che il modello possa essere riporodotto.
Il parametro stratify serve per mantenere la proporzione delle classi nel dataset di addestramento e in quello di test.
"""


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train) # calcola la media e la deviazione standard delle caratteristiche del dataset di addestramento
X_train_std = sc.transform(X_train) # standardizza il dataset di addestramento
X_test_std = sc.transform(X_test) # standardizza il dataset di test
"""
La standardizzazione delle caratteristiche è una pratica comune in ML, poichè molti algoritmi di ML e di ottimizzazione richiedono anche la riduzione in scala delle caratteristiche per grarantire prestazioni ottimali.
"""


from sklearn.svm import SVC

svm = SVC(kernel='linear', C=10, random_state=1) 
svm.fit(X_train_std, y_train)
"""
Dopo l'addestramento del modello, possiamo fare delle previsioni sul dataset di test. 
"""

X_combinate = np.vstack((X_train_std, X_test_std)) # unisce il dataset di addestramento e di test
y_combinate = np.hstack((y_train, y_test)) # unisce le etichette del dataset di addestramento e di test

from plot_decision_region import plot_decision_region

plt = plot_decision_region(X_combinate, y_combinate, classifier=svm, test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('./ppn.png', dpi=300)
plt.show()


