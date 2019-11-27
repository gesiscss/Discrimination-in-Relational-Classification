############################################
# System dependencies
############################################
import numpy as np

############################################
# Class
############################################
class RandomEdges(object):

    def __init__(self, G, pseeds):
        '''
        Initializes the random node sampling object
        - G: global network
        - pseeds: fraction of nodes to sample
        '''
        self.G = G
        self.pseeds = pseeds        
        self.nseeds = int(round(self.G.number_of_nodes() * pseeds))
        np.random.seed(None)
        
    def extract_subgraph(self, **kwargs):
        '''
        Creates a subgraph from G based on the sampling technique
        '''
        edges = self.get_random_edges()
        return self.G.edge_subgraph(edges).copy()
    
    def get_random_edges(self):
        '''
        Randomly selects edges until covering (pseeds * N) nodes
        returns a list of nodes
        '''
        edges = list(self.G.edges())
        np.random.shuffle(edges)
        seeds = set()
        counter = 0
        while len(seeds) < self.nseeds:
            seeds |= set(edges[counter])
            counter += 1
        return edges[:counter]
    
        
            