############################################
# System dependencies
############################################
import pandas as pd
import numpy as np
from collections import Counter

############################################
# Class
############################################
class nBC(object):

    def __init__(self):
        '''
        Initializes the network-only Bayes object.
        '''
        self.Gseeds = None
        self.conprob = None
        
    def learn(self, Gseeds, smoothing=True, weight=False):
        '''
        Learns the conditional probabilities from the training sample subgraph.
        - Gseeds: training sample subgraph
        - smoothing: whether or not to use Laplace smoothing
        - weight: whether or not use edge weights
        '''
        self.Gseeds = Gseeds
        
        # measuring probabilities based on Bayes theorem
        self.condprob = pd.DataFrame(index=self.Gseeds.graph['labels'], 
                                columns=self.Gseeds.graph['labels'],
                                data=np.zeros((len(self.Gseeds.graph['labels']),len(self.Gseeds.graph['labels']))))
        
        # counting edges (faster than traversing nodes^2)
        edge_counts = Counter([(self.Gseeds.node[edge[0]][self.Gseeds.graph['class']], 
                                self.Gseeds.node[edge[1]][self.Gseeds.graph['class']]) for edge in self.Gseeds.edges()])
        
        # exact edge counts
        for k,v in edge_counts.items():
            s,t = k
            self.condprob.loc[s,t] = v
        
        # if undirected correct for same-class links (times 2), different-class link both ways
        if not self.Gseeds.is_directed():
            labels = self.Gseeds.graph['labels']
            self.condprob.loc[labels[0],labels[0]] *= 2
            self.condprob.loc[labels[1],labels[1]] *= 2
            tmp = self.condprob.loc[labels[0],labels[1]] + self.condprob.loc[labels[1],labels[0]]
            self.condprob.loc[labels[0],labels[1]] = tmp
            self.condprob.loc[labels[1],labels[0]] = tmp
            
        # Laplace smoothing
        if smoothing:
            self.condprob += 1
        
        # normalize
        self.condprob = self.condprob.div(self.condprob.sum(axis=1), axis=0)

    def predict(self, ego, ni, prior):
        '''
        Predicts the new posterior for node ni given its ego network.
        - ego: ego network centered in ni
        - ni: ego node
        - prior: class prior
        returns ci
        '''
        
        # class prior
        ci = prior.copy()
        
        # neighbors counting
        for nj in ego[ni]:
            xj = ego.node[nj][ego.graph['class'] if ego.node[nj]['seed'] else 'xi']
            ci *= self.condprob.loc[:,xj].copy()
            
        # normalize
        ci /= ci.sum()
        
        return ci
    
    def info(self):
        '''
        Prints the conditional probabilities
        '''
        print(self.condprob)