############################################
# System dependencies
############################################
import networkx as nx

############################################
# Local dependencies
############################################
from org.gesis.sampling.random_nodes import RandomNodes
from org.gesis.sampling.random_neighbors import RandomNeighbors
from org.gesis.sampling.random_edges import RandomEdges
from org.gesis.sampling.partial_crawls import PartialCrawls
from org.gesis.sampling.degree_rank import DegreeRank
from utils.estimator import get_min_degree
from utils.estimator import get_minority_fraction
from utils.estimator import get_similitude
from utils.estimator import get_homophily
from utils.estimator import get_degrees

############################################
# Constants
############################################
RANDOM_NODES = "nodes"
RANDOM_EDGES = "nedges"
DEGREE_RANK = "degree"
RANDOM_NEIGHBORS = "neighbors"
PARTIAL_CRAWLS = "partial_crawls"

############################################
# Class
############################################
class Sampling(object):

    def __init__(self, method, G, pseeds, epoch):
        '''
        Initializes the sampling object
        - method: sampling method
        - G: global network
        - pseeds: fraction of seeds to sample
        '''
        self.method = method
        self.G = G
        self.pseeds = pseeds
        self.epoch = epoch
        self.Gseeds = None
        
    def extract_subgraph(self, **kwargs):
        '''
        Creates a new instance of the respective sampling method, and calls its respective extract_subgraph method.
        '''
        if self.pseeds <= 0 or self.pseeds >= 1:
            raise Exception("pseeds value exception: {}".format(self.pseeds))
            
        if self.method == RANDOM_NODES:
            self.Gseeds = RandomNodes(G=self.G, pseeds=self.pseeds).extract_subgraph()
        elif self.method == RANDOM_NEIGHBORS:
            self.Gseeds = RandomNeighbors(G=self.G, pseeds=self.pseeds).extract_subgraph()
        elif self.method == RANDOM_EDGES:
            self.Gseeds = RandomEdges(G=self.G, pseeds=self.pseeds).extract_subgraph()
        elif self.method == DEGREE_RANK:
            self.Gseeds = DegreeRank(G=self.G, pseeds=self.pseeds).extract_subgraph()
        elif self.method == PARTIAL_CRAWLS:
            self.Gseeds = PartialCrawls(G=self.G, pseeds=self.pseeds, sn=kwargs['sn']).extract_subgraph()
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
        self.Gseeds.graph['m'] = get_min_degree(self.Gseeds)
        self.Gseeds.graph['N'] = self.Gseeds.number_of_nodes()
        self.Gseeds.graph['B'] = get_minority_fraction(self.Gseeds)
        self.Gseeds.graph['H'] = get_homophily(self.Gseeds)
        self.Gseeds.graph['h'] = get_similitude(self.Gseeds)
        self.Gseeds.graph['e'] = self.Gseeds.number_of_edges()
        k, km, kM = get_degrees(self.Gseeds)
        self.Gseeds.graph['k'] = k
        self.Gseeds.graph['km'] = km
        self.Gseeds.graph['kM'] = kM
        self.Gseeds.graph['epoch'] = self.epoch

        del (self.Gseeds.graph['n'])
        del (self.Gseeds.graph['b'])
        del (self.Gseeds.graph['min_degree'])

    def info(self):
        '''
        Prints a summary of the training sample subgraph, including its attributes.
        '''
        print(nx.info(self.Gseeds))
        print(self.Gseeds.graph)
        
            