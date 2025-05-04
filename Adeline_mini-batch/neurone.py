import numpy as np 


class AdalineSGD(object):
    """ADAptive LInear NEuron classifier.

    Parameters
    ----------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.
    shuffle : bool (default: False)
        Shuffles training data every epoch if True to prevent cycles.
    random_state : int (default: None)
        Random number generator seed for random weight initialization.

    Attributes
    ----------
    w_ : 1d-array
        Weights after fitting.
    cost_ : list
        Sum of squares cost function avaraged over all training samples in each epoch.

    """
    
    def __init__(self, eta=0.01, n_iter=50, shuffle=False, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.shuffle = shuffle
        self.random_state = random_state
        self.w_initialized = False
        
    def fit(self, X, y):
        """Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        self : object
            Returns self.

        """
        
        self._initialize_weights(X.shape[1])
        self.cost_ = []
        # Main loop
        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost)
            
        return self
    
    def partial_fit(self, X, y):
        """Fit training data without reinitializing the weights.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        self : object
            Returns self.

        """
        
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
            
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
            
        return self
    
    def _shuffle(self, X, y):
        """Shuffle training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        X : {array-like}, shape = [n_samples, n_features]
            Shuffled training vectors.
        y : array-like, shape = [n_samples]
            Shuffled target values.

        """
        
        r = self.rgen.permutation(len(y))
        return X[r], y[r]
    
    def _initialize_weights(self, m):
        """Initialize weights to zeros.

        Parameters
        ----------
        m : int
            Number of features.

        """
        
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=m + 1)
        self.w_initialized = True
        
    def _update_weights(self, xi, target):
        """Apply Adaline learning rule to update the weights.

        Parameters
        ----------
        xi : {array-like}, shape = [n_features]
            Single training sample.
        target : float
            Target value.

        Returns
        -------
        cost : float
            Sum of squares cost function.

        """
        
        output = self.activation(self.net_input(xi))
        error = (target - output)
        cost = 0.5 * error ** 2
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        
        return cost
    
    def activation(self, X):
        """Compute linear activation function.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Input data.

        Returns
        -------
        linear_output : {array-like}, shape = [n_samples]
            Linear activation function output.

        """
        
        return X
    
    def net_input(self, X):
        """Calculate net input.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Input data.

        Returns
        -------
        net_input : {array-like}, shape = [n_samples]
            Net input.

        """
        
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict(self, X):
        """Return class label after unit step.
        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Input data.
        Returns
        -------
        class_label : {array-like}, shape = [n_samples]
            Predicted class label.
        """         
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)
    
    
    
    
    