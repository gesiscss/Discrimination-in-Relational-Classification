############################################
# System dependencies
############################################
import pandas as pd
import numpy as np
from collections import Counter

############################################
# Class
############################################
class ClassPrior(object):

    def __init__(self, Gseeds):
        '''
        Initializes the class prior object
        - Gseeds: training sample
        '''
        self.Gseeds = Gseeds        
        
    def learn(self):
        '''
        Learns the class prior from the training sample.
        Simply proportion of nodes in each class.
        returns prior 
        '''
        tmp = Counter([self.Gseeds.node[n][self.Gseeds.graph['class']] for n in self.Gseeds.nodes()])        
        minority = tmp.most_common(2)[1]
        majority = tmp.most_common(2)[0]
        
        prior = pd.Series(index=[majority[0],minority[0]])
        prior.loc[majority[0]] = majority[1]/self.Gseeds.number_of_nodes()
        prior.loc[minority[0]] = minority[1]/self.Gseeds.number_of_nodes()
        
        return prior               
        
            