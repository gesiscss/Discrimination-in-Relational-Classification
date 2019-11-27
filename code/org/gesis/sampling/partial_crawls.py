############################################
# System dependencies
############################################
import numpy as np
import operator

############################################
# Class
############################################
class PartialCrawls(object):

    def __init__(self, G, pseeds, sn):
        '''
        Initializes the random node sampling object
        - G: global network
        - pseeds: fraction of nodes to sample
        '''
        self.G = G
        self.S = None
        self.pseeds = pseeds     
        self.sn = sn
        self.nseeds = int(round(self.G.number_of_nodes() * pseeds))
        np.random.seed(None)
        
    def extract_subgraph(self, **kwargs):
        '''
        Creates a subgraph from G based on the sampling technique
        '''
        self.create_super_node()
        edges = self.get_edges_from_tours()
        return self.G.edge_subgraph(edges).copy()
    
    def create_super_node(self):
        '''
        Randomly selects (sn * N) nodes
        returns a list of nodes
        '''
        nodes = list(self.G.nodes())
        np.random.shuffle(nodes)
        super_node_size = int(round(self.G.number_of_nodes() * self.sn))
        self.S = nodes[:super_node_size]
    
    def get_edges_from_tours(self):
        '''
        Sorts nodes in super node proportional to their edges outside the super node.
        Perfoms random walks from/to super node until collecting (pseeds * N) nodes
        returns a list od edges
        '''
        # proportional to the number of edges out of the super node
        sorted_S = {ni:len([nj for nj in self.G.neighbors(ni) if nj not in self.S]) for ni in self.G.nodes()}
        sorted_S = sorted(sorted_S.items(), key=operator.itemgetter(1), reverse=True)
        sorted_S = [n[0] for n in sorted_S]
        
        # tours
        edges = []
        sampled_nodes = set()
        for vi in sorted_S:
            vj = np.random.choice(list(self.G.neighbors(vi)), 1)[0] # random neighbor
            while vj not in self.S and len(sampled_nodes) < self.nseeds: 
                edges.append((vi,vj))
                sampled_nodes |= set([vi,vj])
                vi = vj
                vj = np.random.choice(list(self.G.neighbors(vi)), 1)[0] # random neighbor
                
        return edges
            