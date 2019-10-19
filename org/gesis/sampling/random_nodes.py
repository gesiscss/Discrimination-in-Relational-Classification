############################################
# System dependencies
############################################
import numpy as np

############################################
# Class
############################################
class RandomNodes(object):

    def __init__(self, G, pseeds):
        '''
        Initializes the random node sampling object
        - G: global network
        - pseeds: fraction of nodes to sample
        '''
        self.G = G
        self.pseeds = pseeds        
        self.nseeds = int(round(self.G.number_of_nodes() * pseeds))
        
    def extract_subgraph(self, **kwargs):
        '''
        Creates a subgraph from G based on the sampling technique
        '''
        nodes = self.get_random_nodes()
        return self.G.subgraph(nodes).copy()
    
    def get_random_nodes(self):
        '''
        Randomly selects (pseeds * N) nodes
        returns a list of nodes
        '''
        nodes = list(self.G.nodes())
        np.random.shuffle(nodes)
        return nodes[:self.nseeds]
    
        
            