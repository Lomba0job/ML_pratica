import numpy as np 

class ADalineGD(object):
    """
    ADAptive LInear NEuron classifier

    Parametri
    ---------
    eta (float) : Learning rate[ 0.0 - 1.0]
    n_inter (int) : Passes over ther trainig dataset
    random_state (int) : Random Number
    
    
    Attributi
    --------
    w_ (1d-array): somma dei quadrati per ogni epoca
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
        self.cost_  = []
        
        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = y - output
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self
    
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def activation(self, X):
        return X
    
    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)
        