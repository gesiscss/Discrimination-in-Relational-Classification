############################################
# System dependencies
############################################

############################################
# Local dependencies
############################################
from org.gesis.local.class_prior import ClassPrior

############################################
# Constants
############################################
CLASS_PRIOR = "prior"

############################################
# Class
############################################
class Local(object):

    def __init__(self, method):
        '''
        Initializes the local model object
        - method: local model method
        '''
        self.method = method        
        self.prior = None
        
    def learn(self, Gseeds):
        '''
        Creates a new instance of the respective local model method
        - Gseeds: training sample subgraph
        '''
        if self.method == CLASS_PRIOR:
            self.prior = ClassPrior(Gseeds).learn()
        else:
            raise Exception("local model does not exist: {}".format(self.method))
               
    def info(self):
        '''
        Print the class prior
        '''
        print(self.prior)        
        
            