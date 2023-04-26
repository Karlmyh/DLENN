import numpy as np
import math
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
rom sklearn.model_selection import KFold

class DLENN(object):
    def __init__(
        self,
        weighting_scheme = "extrapolated",
        n_machine = 1,
        order = 1,
        lamda = 1.0,
        k = 2,
        random_state = 1,
    ):
        
        self.weighting_scheme = weighting_scheme
        self.n_machine_per_order = n_machine_per_order
        self.order = order
        self.lamda = lamda
        self.k = k
        self.random_state = random_state
        
        self.order_constant_vec = np.array([i + 1 for i in range(self.order)] )
        self.k_vec = np.repeat(self.order_constant_vec * self.k, self.n_machine_per_order)

    def fit(self, X, y):
        
        kfolder = KFold(n_splits = self.n_machine_per_order * self.order , random_state = self.random_state, shuffle = True)

        self.regressor_vec = []
        self.dim = X_vec[0].shape[1]
        
        for i, (_, test_index) in enumerate(kfolder.split(X)):
            self.regressor_vec.append( 
                KNeighborsRegressor(n_neighbors = self.k_vec[i]).fit(X[test_index, :], y[test_index])
            )
        
        if self.weighting_scheme == "uniform":
            self.weights = np.repeat(1 / len(self.regressor_vec), len(self.regressor_vec))
        elif self.weighting_scheme == "extrapolated":
            r_mat = np.array([ [k**(- 2 * i / self.dim) for i in range(self.order)] for k in self.k_vec])
            self.weights = (np.linalg.inv(r_mat.T @ r_mat + self.lamda * np.eye(self.smooth_order)) @ r_mat.T)[0]
            if self.weights.sum():
                self.weights = self.weights / self.weights.sum()
            else:
                self.weights[self.weights < 0] = 0
                self.weights = self.weights / self.weights.sum()
                
        return self
    
    
    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        out = dict()
        for key in ['n_machine',"lamda","weighting_scheme","order","k"]:
            value = getattr(self, key, None)
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out
    
    
    def set_params(self, **params):
        """Set the parameters of this estimator.

        The method works on simple estimators as well as on nested objects
        (such as pipelines). The latter have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.

        Returns
        -------
        self
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)


        for key, value in params.items():
            if key not in valid_params:
                raise ValueError('Invalid parameter %s for estimator %s. '
                                 'Check the list of available parameters '
                                 'with `estimator.get_params().keys()`.' %
                                 (key, self))
            setattr(self, key, value)
            valid_params[key] = value

        return self


    def predict(self, X):
        
        prediction_by_nodes = np.array([regressor.predict(X) for regressor in self.regressor_vec])
        return self.weights @ prediction_by_nodes 
        
        
    
    
    def score(self, X, y):
        return - mean_squared_error(self.predict(X), y)
            
    

    

