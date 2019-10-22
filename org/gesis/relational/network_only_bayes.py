############################################
# System dependencies
############################################
import pandas as pd
import numpy as np
from collections import Counter
from scipy.special import logsumexp

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
        https://stats.stackexchange.com/questions/105602/example-of-how-the-log-sum-exp-trick-works-in-naive-bayes
        Predicts the new posterior for node ni given its ego network.
        - ego: ego network centered in ni
        - ni: ego node
        - prior: class prior
        returns ci
        '''
        return self._predict(ego, ni, prior)
    
    def _predict_log(self, ego, ni, prior):
        
        if len(list(ego.neighbors(ni))) == 0:
            return prior.copy()
            
        # neighbors counting (conditional probabilities)
        likelihood = pd.Series([0,0], index=prior.index)
        for nj in ego[ni]:
            xj = ego.node[nj][ego.graph['class'] if ego.node[nj]['seed'] else 'xi']
            likelihood = likelihood.add( np.log(self.condprob.loc[:,xj].copy()) )
        likelihood = likelihood.subtract( logsumexp(likelihood) )
        
        # class prior
        prior = np.log( prior.copy() )
        
        # posterior
        ci = prior.add( likelihood )
        
        # normalizing
        ci = ci.subtract( logsumexp(ci) )
        
        # eliminating log
        ci = np.exp( ci )

        return ci
    
    def _predict(self, ego, ni, prior):
        
        if len(list(ego.neighbors(ni))) == 0:
            return prior.copy()
            
        # neighbors counting (conditional probabilities)
        ci = pd.Series([1,1], index=prior.index)
        for nj in ego[ni]:
            xj = ego.node[nj][ego.graph['class'] if ego.node[nj]['seed'] else 'xi']
            ci *= self.condprob.loc[:,xj].copy()
        ci /= ci.sum()
        
        # class prior
        ci *= prior.copy()
        
        # normalizing
        ci /= ci.sum()
        
        return ci
    
    def info(self):
        '''
        Prints the conditional probabilities
        '''
        print(self.condprob)