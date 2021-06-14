############################################
# System dependencies
############################################
import numpy as np
import networkx as nx

############################################
# Class
############################################
class Relaxation(object):

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
        
    def predict(self):
        '''
        Predicts the new posterior and class label of nodes using relaxation labeling.
        Values are store per node as node attributes ci and xi respectively
        '''
        T = np.arange(0,99,1)
        k = 1.0
        alpha = 0.99

        unlabeled = [n for n in self.G.nodes() if not self.G.node[n]['seed']]
        
        # 1. Initialize with prior (step 1)
        for n in unlabeled:
            self.G.node[n]['ci'] = self.local_model.prior
            self.G.node[n]['xi'] = np.random.choice(a=self.G.graph['labels'], p=self.G.node[n]['ci'].values)
        
        # 2. Estimate xi by applying the relational model T times (steps 2, ..., 100)
        beta = k
        for t in T:
            beta *= alpha
            nx.set_node_attributes(G=self.G, name='ci', values={n:self._update_ci(n, beta) for n in unlabeled})
            nx.set_node_attributes(G=self.G, name='xi', values={n:self._update_xi(n) for n in unlabeled})
                                   
    def _update_ci(self, n, beta):
        '''
        Computes the new posterior probability for node n
        - n: ego node
        - beta: learning rate
        returns ci 
        '''
        _prev = (1-beta) * self.G.node[n]['ci']
        _new = beta * self.relational_model.predict(nx.ego_graph(self.G, n), n, self.local_model.prior)
        return _new + _prev
                
    def _update_xi(self, n):
        '''
        Computes the new class label for node n
        - n: ego node
        returns xi 
        '''
        if abs(self.G.node[n]['ci'].values[0] - 0.5) < 0.1:
             return np.random.choice(a=self.G.graph['labels'], p=self.G.node[n]['ci'].values)
        return self.G.node[n]['ci'].idxmax()


        