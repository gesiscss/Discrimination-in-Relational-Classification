############################################
# Local dependencies
############################################
from utils.estimator import nBC_learn

############################################
# System dependencies
############################################
import pandas as pd
import numpy as np
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
        self.condprob = nBC_learn(self.Gseeds, smoothing, weight)

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