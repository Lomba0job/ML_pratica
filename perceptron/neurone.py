import numpy as np 

class Perceptron(object):
    
    """
    Parametri
    ---------
    eta (float) : Learning rate[ 0.0 - 1.0]
    n_inter (int) : Passes over ther trainig dataset
    random_state (int) : Random Number
    
    
    Attributi
    --------
    w_ (1d-array): pesi dopo il fitting
    errors_ (list) : lista degli errori in ogni passo
        
    """
    
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        
    def fit(self, X, y):
        """fit training data

        Args:
            X (array-like): shape = [n_examples, n_fetures] exampples and n_features
            y (arrai-like): shape = [n_examples] Target
            
        Returns:
            self: object
        """        
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []
        
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors  += int(update != 0.0)
            self.errors_.append(errors)
        return self
    
    def net_input(self, X):
        return np.dot(X, self.w_[1:] + self.w_[0])
    
    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)
                