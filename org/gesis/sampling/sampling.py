############################################
# System dependencies
############################################
import networkx as nx

############################################
# Local dependencies
############################################
from org.gesis.sampling.random_nodes import RandomNodes

############################################
# Constants
############################################
RANDOM_NODES = "nodes"
RANDOM_EDGES = "nedges"
DEGREE_RANKING = "degree"
RANDOM_NEIGHBORS = "neighbors"
PARTIAL_CRAWLS = "partial_crawls"

############################################
# Class
############################################
class Sampling(object):

    def __init__(self, method, G, pseeds):
        '''
        Initializes the sampling object
        - method: sampling method
        - G: global network
        - pseeds: fraction of seeds to sample
        '''
        self.method = method
        self.G = G
        self.pseeds = pseeds
        self.Gseeds = None
        
    def extract_subgraph(self, **kwargs):
        '''
        Creates a new instance of the respective sampling method, and calls its respective extract_subgraph method.
        '''
        if self.pseeds <= 0 or self.pseeds >= 1:
            raise Exception("pseeds value exception: {}".format(self.pseeds))
            
        if self.method == RANDOM_NODES:
            self.Gseeds = RandomNodes(self.G, self.pseeds).extract_subgraph()
        else:
            raise Exception("sampling method does not exist: {}".format(self.method))
        
        self.set_graph_metadata()
        
    def set_graph_metadata(self):
        '''
        Updates the training sample sugraph metadata
        '''
        self.Gseeds.graph['pseeds'] = self.pseeds
        self.Gseeds.graph['method'] = self.method
        nx.set_node_attributes(G=self.G, name='seed', values={n:n in self.Gseeds for n in self.G.nodes()})
    
    def info(self):
        '''
        Prints a summary of the training sample subgraph, including its attributes.
        '''
        print(nx.info(self.Gseeds))
        print(self.Gseeds.graph)
        
            