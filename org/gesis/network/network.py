############################################
# System dependencies
############################################
import networkx as nx

############################################
# Local dependencies
############################################
from org.gesis.network.generate_homophilic_graph_symmetric import homophilic_barabasi_albert_graph as BAH

############################################
# Constants
############################################
BARABASI_ALBERT_HOMOPHILY = "BAH"

############################################
# Class
############################################
class Network(object):

    def __init__(self, kind):
        '''
        Initializes the network object
        - kind: type of network
        '''
        self.kind = kind
        self.G = None
        
    def create_network(self, **kwargs):
        '''
        Creates a new instance of the respective network type
        - kwargs: network properties
        '''
        if self.kind == BARABASI_ALBERT_HOMOPHILY:            
            self.G = BAH(N=kwargs['N'], m=kwargs['m'], minority_fraction=kwargs['B'], similitude=kwargs['H'])
        else:
            raise Exception("{}: network type does not exist.".format(kind))
        
        self.set_graph_metadata()
    
    def set_graph_metadata(self):
        '''
        Updates the metadata of the graph as graph properties
        '''
        self.G.graph['name'] = self.kind
           
    def info(self):
        '''
        Prints a summary of the network, including its attributes.
        '''
        print(nx.info(self.G))
        print(self.G.graph)
        
            