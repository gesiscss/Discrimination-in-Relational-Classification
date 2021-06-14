############################################
# System dependencies
############################################
import numpy as np
import networkx as nx
import pandas as pd

############################################
# Class
############################################
class RelaxationLINK(object):

    def __init__(self, G, local_model, relational_model):
        '''
        Initializes the relaxation labeling object
        - G: global network
        - local_model: instance of LC
        - relational_model: instance of RC
        '''
        self.G = G
        self.local_model = local_model
        self.relational_model = relational_model
        self.xi = None
        self.ci = None

    def predict(self):
        '''
        Predicts the new posterior and class label of nodes using relaxation labeling.
        Values are store per node as node attributes ci and xi respectively
        '''
        T = np.arange(0,99,1)
        k = 1.0
        alpha = 0.99

        # 1. Initialize with prior (step 1)
        n_test = self.relational_model.test_index.shape[0]
        n_features = self.local_model.prior.shape[0]
        # print(self.local_model.prior.values)
        # print(self.relational_model.test_index.shape)
        # print(self.local_model.prior.shape)

        self.ci = np.ones((n_test, n_features)) * self.local_model.prior.values
        self.xi = np.random.choice(a=self.G.graph['labels'], p=self.local_model.prior, size=self.ci.shape[0])
        
        # 2. Estimate xi by applying the relational model T times (steps 2, ..., 100)
        beta = k
        for t in T:
            beta *= alpha
            self.ci = self._update_ci(beta) # new pior
            self.xi = self._update_xi() # new class label

        for i,n in enumerate(self.relational_model.test_nodes):
            self.G.node[n]['ci'] = pd.Series(self.ci[i], index=self.G.graph['labels'])
            self.G.node[n]['xi'] = self.xi[i]

    def _update_ci(self, beta):
        '''
        Computes the new posterior probability for node n
        '''
        _prev = (1-beta) * self.ci
        _new = beta * self.relational_model.predict()
        return _new + _prev
                
    def _update_xi(self):
        '''
        Computes the new class label xi for all nodes ni in the test set
        returns xi 
        '''
        return [np.random.choice(a=self.G.graph['labels'], p=self.ci[i]) if abs(self.ci[i][0]-0.5)<0.1 else self.G.graph['labels'][self.ci[i].argmax()] for i,j in enumerate(self.relational_model.test_index)]



        