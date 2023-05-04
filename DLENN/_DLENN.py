import numpy as np
import math
from sklearn.neighbors import KNeighborsRegressor
from ._regressor import RadiusNeighborsRegressor
from sklearn.neighbors import KDTree
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

class DLENN(object):
    def __init__(
        self,
        weighting_scheme = "extrapolated",
        n_machine_per_order = 1,
        order = 1,
        lamda = 1.0,
        k = 2,
        random_state = 1,
        increment = 0.1,
        use_radius = True,
    ):
        
        print("00")
        
        self.weighting_scheme = weighting_scheme
        self.n_machine_per_order = n_machine_per_order
        self.order = order
        self.lamda = lamda
        self.k = k
        self.random_state = random_state
        self.increment = increment
        self.use_radius = use_radius
        
        print("01")
        
    def fit(self, X, y):
        
        
        print("1")
        kfolder = KFold(n_splits = self.n_machine_per_order * (self.order + 1), random_state = self.random_state, shuffle = True)

        print("2")
        
        self.regressor_vec = []
        self.dim = X.shape[1]
        
        if self.weighting_scheme == "uniform":
            self.weights = np.repeat(1 / (self.order + 1) / self.n_machine_per_order, int((self.order + 1) * self.n_machine_per_order))
            for i, (_, test_index) in enumerate(kfolder.split(X)):
                self.regressor_vec.append( 
                    KNeighborsRegressor(n_neighbors = min(self.k, test_index.shape[0]) ).fit(X[test_index, :], y[test_index])
                )
                
            print("3")
            
        
        
        elif self.weighting_scheme == "extrapolated":
            
            self.order_constant_vec = np.array([self.increment * i + 1 for i in range(self.order + 1)] )
            
            ######## if use radius prediction ########
            if self.use_radius:
#                 # compute average k distance 
#                 avg_dist = 0
#                 max_dist = 0
#                 for i, (_, test_index) in enumerate(kfolder.split(X)):
#                     tree = KDTree(X[test_index, :])
#                     distance_vec,_ = tree.query(X[test_index, :], self.k + 1)
#                     avg_dist += distance_vec[:,-1].mean()
#                     if distance_vec[:,-1].mean() > max_dist:
#                         max_dist = distance_vec[:,-1].mean()
#                 avg_dist /= self.n_machine_per_order * (self.order + 1)
                
                # compute the radius vector
                self.radius_vec = np.repeat((self.order_constant_vec ).astype(int), self.n_machine_per_order)
               
                
                # get regressor sequence
                for i, (_, test_index) in enumerate(kfolder.split(X)):
                    regressor = RadiusNeighborsRegressor(n_neighbors = self.k,
                                                         ratio = self.radius_vec[i],
                                                        ).fit(X[test_index, :], y[test_index])
                    self.regressor_vec.append( regressor )
                
                # get r matrix
                r_mat = np.array([ [r**( 2 * i ) for i in range(self.order + 1)] for r in self.radius_vec])
            ######## if use neighbor number prediction ######## 
            else:
                # compute the neighbor number vector
                self.k_vec = np.repeat((self.order_constant_vec * self.k).astype(int), self.n_machine_per_order)
                
                # get regressor sequence
                for i, (_, test_index) in enumerate(kfolder.split(X)):
                    regressor = KNeighborsRegressor(n_neighbors = min(self.k_vec[i], 
                                                    test_index.shape[0])).fit(X[test_index, :], y[test_index])
                    self.regressor_vec.append( regressor )
                    
                # get r matrix
                r_mat = np.array([ [k**( 2 * i / self.dim) for i in range(self.order + 1)] for k in self.k_vec])
    
            # compute weights according to r matrix
            self.weights = (np.linalg.inv(r_mat.T @ r_mat + self.lamda * np.eye(self.order + 1)) @ r_mat.T)[0]
            if self.weights.sum():
                self.weights = self.weights / self.weights.sum()
            else:
                self.weights[self.weights < 0] = 0
                self.weights = self.weights / self.weights.sum()
                
            print("4")
            
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
        for key in ['n_machine_per_order',"lamda","weighting_scheme","order","k",
                   "increment", "use_radius"]:
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
            
    

    

