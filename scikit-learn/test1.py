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


from sklearn.linear_model import Perceptron

ppn = Perceptron(eta0=0.1, random_state=1) # eta0 è il tasso di apprendimento
ppn.fit(X_train_std, y_train) # addestra il modello sul dataset di addestramento
"""
Dopo l'addestramento del modello, possiamo fare delle previsioni sul dataset di test. 
"""

y_pred = ppn.predict(X_test_std) # predice le etichette del dataset di test
print("Misclassified samples: ", (y_test != y_pred).sum()) # conta il numero di campioni classificati in modo errato

from sklearn.metrics import accuracy_score
print("Accuracy: ", accuracy_score(y_test, y_pred)) # calcola l'accuratezza del modello
"""
L'accuratezza è una misura di quanto il modello è preciso nel classificare i dati.
In questo caso, l'accuratezza è del 93.33%, il che significa che il modello ha classificato correttamente il 93.33% dei campioni del dataset di test.
"""

