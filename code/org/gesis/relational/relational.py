############################################
# Local dependencies
############################################
from org.gesis.relational.network_only_bayes import nBC
from org.gesis.relational.link import LINK

############################################
# Constants
############################################
NETWORK_ONLY_BAYES = "nBC"
NETWORK_ONLY_LINK_BASED = "LINK"

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
        elif self.method == NETWORK_ONLY_LINK_BASED:
            return LINK()
        else:
            raise Exception("relational model does not exist: {}".format(self.method))
