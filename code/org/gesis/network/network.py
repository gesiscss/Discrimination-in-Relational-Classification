############################################
# System dependencies
############################################
import networkx as nx

############################################
# Local dependencies
############################################
from org.gesis.network.generate_homophilic_graph_symmetric import homophilic_barabasi_albert_graph as BAH
from utils.estimator import get_similitude
from utils.estimator import get_degrees
from utils.io import load_gpickle

############################################
# Constants
############################################
BARABASI_ALBERT_HOMOPHILY = "BAH"

############################################
# Class
############################################
class Network(object):

    def __init__(self, kind=None):
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

            h = get_similitude(self.G)
            k,km,kM = get_degrees(self.G)

            self.G.graph['kind'] = self.kind
            self.G.graph['fullname'] = "{}-N{}-m{}-B{}-H{}-i{}-x{}-h{}-k{}-km{}-kM{}".format(self.kind,
                                                                                             kwargs['N'],
                                                                                             kwargs['m'],
                                                                                             kwargs['B'],
                                                                                             kwargs['H'],
                                                                                             1 if 'i' not in kwargs else kwargs['i'],
                                                                                             1 if 'x' not in kwargs else kwargs['x'],
                                                                                             round(h,1),
                                                                                             round(k,1),
                                                                                             round(km,1),
                                                                                             round(kM,1))
        else:
            raise Exception("{}: network type does not exist.".format(self.kind))

    def load(self, datafn, ignoreInt=None):
        '''
        Loads gpickle graph into G
        :param datafn:
        :return:
        '''
        self.G = load_gpickle(datafn)
        self._validate(ignoreInt)

    def _validate(self, ignoreInt=None):

        # 1. ignoreInt
        if ignoreInt is not None:
            to_remove = [n for n in self.G.nodes() if self.G.node[n][self.G.graph['class']] == ignoreInt]
            self.G.remove_nodes_from(to_remove)
            self.G.graph['ignoreInt'] = ignoreInt

        # degree 0
        to_remove = [n for n in self.G.nodes() if self.G.degree(n) == 0]
        self.G.remove_nodes_from(to_remove)
        self.G.graph['min_degree'] = min([d for n,d in self.G.degree()])


    def info(self):
        '''
        Prints a summary of the network, including its attributes.
        '''
        print(nx.info(self.G))
        print(self.G.graph)
        
            