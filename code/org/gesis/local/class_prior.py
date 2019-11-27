############################################
# Local dependencies
############################################
from utils.estimator import prior_learn

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
        prior = prior_learn(self.Gseeds)
        return prior               
        
            