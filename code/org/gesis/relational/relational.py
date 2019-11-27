############################################
# Local dependencies
############################################
from org.gesis.relational.network_only_bayes import nBC

############################################
# Constants
############################################
NETWORK_ONLY_BAYES = "nBC"

############################################
# Class
############################################
class Relational(object):
    
    def __init__(self, method):
        '''
        Initializes the relational model algorithm
        - method: relational model method
        '''
        self.method = method
        
    def get_model(self):
        '''
        Creates a new instance of the respective relational model algorithm
        '''
        if self.method == NETWORK_ONLY_BAYES:
            return nBC()
        else:
            raise Exception("relational model does not exist: {}".format(self.method))
