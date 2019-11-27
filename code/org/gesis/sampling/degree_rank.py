############################################
# System dependencies
############################################
import numpy as np
import operator

############################################
# Class
############################################
class DegreeRank(object):

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
        nodes = self.get_ranked_nodes()
        return self.G.subgraph(nodes).copy()
    
    def get_ranked_nodes(self):
        '''
        Sorts nodes by degree descending, and returns only (pseeds * N) nodes
        returns a list of nodes
        '''
        nodes = {n:self.G.degree(n) for n in self.G.nodes()}
        nodes = sorted(nodes.items(), key=operator.itemgetter(1), reverse=True)
        return [n[0] for n in nodes][:self.nseeds]
    
        
            