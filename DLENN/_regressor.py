import numpy as np
import math

from sklearn.neighbors import KDTree


class RadiusNeighborsRegressor(object):
    def __init__(
        self,
        n_neighbors = 2,
        ratio = 1,
    ):
        
        self.n_neighbors = n_neighbors
        self.ratio = ratio
        
        
    def fit(self, X, y):
        
        self.tree_ = KDTree(X)
        self.y = y          
                
            
        return self
    
   

    def predict(self, X):
        distance_vec, _ = self.tree_.query(X, self.n_neighbors )
        distance_vec = distance_vec[:,-1].ravel()
        
        prediction = np.zeros(X.shape[0])
        for idx, distance in enumerate(distance_vec):
            idx_in_sample = self.tree_.query_radius(X[idx,:].reshape(1,-1), r = self.ratio * distance)[0]
            prediction[idx] = self.y[idx_in_sample].mean()
        return prediction
        
        
    
    
 
    

    

