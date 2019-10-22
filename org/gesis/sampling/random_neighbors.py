############################################
# System dependencies
############################################
import numpy as np

############################################
# Class
############################################
class RandomNeighbors(object):

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
        edges = self.get_random_nodes_and_neighbors()
        return self.G.edge_subgraph(edges).copy()
    
    def get_random_nodes_and_neighbors(self):
        '''
        Shuffles the list of nodes, and selects edges of selected node until (pseeds * N) nodes
        returns a list of nodes
        '''
        nodes = list(self.G.nodes())
        np.random.shuffle(nodes)
        edges = []
        seeds = set()
        for ni in nodes:
            
            if len(seeds) >= self.nseeds:
                break
                
            neighbors = list(self.G.neighbors(ni))
            
            if len(seeds)+len(neighbors)+1 > self.nseeds:
                np.random.shuffle(neighbors)
                n = self.nseeds-len(seeds)-1
                if n>0:
                    neighbors = neighbors[:n]
                else:
                    break
                
            edges.extend([(ni,nj) for nj in neighbors])
            seeds |= set(neighbors) | set([ni])
            
        return edges
    
        
            