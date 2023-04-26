import numpy as np
import math
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

class FLENN(object):
    def __init__(
        self,
        portion_vec = [1],
        lamda = 1.0,
        weighting_scheme = "extrapolated",
        smooth_order = 1,
    ):
        self.portion_vec = portion_vec
        self.lamda = lamda
        self.weighting_scheme = weighting_scheme
        self.smooth_order = smooth_order
        
        
     


    def fit(self, X, y):
        
        
        X_vec, y_vec = self.distribute_data(X, y)
        
        assert self.smooth_order <= len(X_vec)
        
        
        
        self.regressor_vec = []
        
        self.dim = X_vec[0].shape[1]
        self.sample_size_vec = [X.shape[0] for X in X_vec]
        for i in range(len(y_vec)):
            potential_vec = [1, 2, 5,10,20,40,60,80,100,120,150,200,250,300]
            params = {"n_neighbors" : [n for n in potential_vec if n < max(2,X_vec[i].shape[0]/2)]}
            cv_regressor = GridSearchCV(estimator = KNeighborsRegressor(), param_grid = params, cv = 3)
            cv_regressor.fit(X_vec[i], y_vec[i])
            self.regressor_vec.append(cv_regressor.best_estimator_)
        
        if self.weighting_scheme == "uniform":
            self.weights = np.repeat(1/len(y_vec), len(y_vec))
        elif self.weighting_scheme == "extrapolated":
            r_mat = np.array([ [size**(- 2 * i / self.dim) for i in range(self.smooth_order)] for size in self.sample_size_vec])
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
        for key in ['portion_vec',"lamda","weighting_scheme","smooth_order"]:
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
        
        
    def distribute_data(self, X, y):
        X_vec = []
        y_vec = []
        
        start_idx = 0
        for i in range(len(self.portion_vec) - 1):
            X_vec.append(X[start_idx : (start_idx + int(self.portion_vec[i] * X.shape[0])), :])
            y_vec.append(y[start_idx : (start_idx + int(self.portion_vec[i] * X.shape[0]))])
            start_idx += int(self.portion_vec[i] * X.shape[0])
        
        X_vec.append(X[start_idx : , :])
        y_vec.append(y[start_idx : ])
        
        return X_vec, y_vec
    
    def score(self, X, y):
        return - mean_squared_error(self.predict(X), y)
            
    

    

