############################################
# System dependencies
############################################
import networkx as nx
import os

############################################
# Local dependencies
############################################
from org.gesis.network.generate_homophilic_graph_symmetric import homophilic_barabasi_albert_graph as BAHsym
from org.gesis.network.homophilic_barabasi_albert_graph_assym import homophilic_barabasi_albert_graph_assym as BAHasym
from utils.estimator import get_similitude
from utils.estimator import get_homophily
from utils.estimator import get_minority_fraction
from utils.estimator import get_average_degrees
from utils.estimator import get_min_degree
from utils.estimator import get_density
from utils.estimator import get_param
from utils.io import load_gpickle
from utils.io import write_gpickle

############################################
# Constants
############################################
BARABASI_ALBERT_HOMOPHILY = "BAH"

############################################
# Class
############################################
class Network(object):

    def __init__(self, kind=None, fit=None):
        '''
        Initializes the network object
        - kind: type of network
        '''
        self.kind = kind
        self.fit = fit
        self.G = None
        
    def create_network(self, **kwargs):
        '''
        Creates a new instance of the respective network type
        - kwargs: network properties
        '''
        sym = False

        if self.kind == BARABASI_ALBERT_HOMOPHILY:

            if kwargs['H'] is not None:
                sym = True
                self.G = BAHsym(N=kwargs['N'], m=kwargs['m'], minority_fraction=kwargs['B'], similitude=kwargs['H'])
            elif kwargs['Hmm'] is not None and kwargs['HMM'] is not None:
                self.G = BAHasym(N=kwargs['N'], m=kwargs['m'], minority_fraction=kwargs['B'], h_mM=1-kwargs['Hmm'], h_Mm=1-kwargs['HMM'])

            h = get_similitude(self.G)
            b = get_minority_fraction(self.G)
            k,km,kM = get_average_degrees(self.G)
            m = get_min_degree(self.G)
            density = get_density(self.G)
            i = 1 if 'i' not in kwargs else kwargs['i']
            x = 1 if 'x' not in kwargs else kwargs['x']

            self.G.graph['kind'] = self.kind
            self.G.graph['N'] = kwargs['N']
            self.G.graph['m'] = kwargs['m']
            self.G.graph['density'] = density
            self.G.graph['B'] = kwargs['B']
            self.G.graph['H'] = kwargs['H']
            self.G.graph['Hmm'] = kwargs['Hmm']
            self.G.graph['HMM'] = kwargs['HMM']
            self.G.graph['i'] = i
            self.G.graph['x'] = x

            self.G.graph['n'] = self.G.number_of_nodes()
            self.G.graph['e'] = self.G.number_of_edges()
            self.G.graph['h'] = h
            self.G.graph['b'] = b
            self.G.graph['min_degree'] = m
            self.G.graph['k'] = k
            self.G.graph['km'] = km
            self.G.graph['kM'] = kM

            prefix = self.kind if self.fit is None else '{}-{}'.format(self.kind, self.fit)
            fullname = "{}-N{}-m{}-B{}-{}-i{}-x{}-h{}-k{}-km{}-kM{}".format(prefix,
                                                                            kwargs['N'],
                                                                            kwargs['m'],
                                                                            kwargs['B'],
                                                                            'H{}'.format(kwargs['H']) if sym else 'Hmm{}-HMM{}'.format(kwargs['Hmm'], kwargs['HMM']),
                                                                            i,
                                                                            x,
                                                                            round(h,1),
                                                                            round(k,1),
                                                                            round(km,1),
                                                                            round(kM,1))

            self.G.graph['fullname'] = fullname
        else:
            raise Exception("{}: network type does not exist.".format(self.kind))

    def load(self, datafn, ignoreInt=None):
        '''
        Loads gpickle graph into G
        :param datafn:
        :return:
        '''
        self.G = load_gpickle(datafn)
        self._validate(datafn, ignoreInt)

    def _validate(self, datafn, ignoreInt=None):

        # 1. ignoreInt
        if ignoreInt is not None:
            to_remove = [n for n in self.G.nodes() if self.G.node[n][self.G.graph['class']] == ignoreInt]
            self.G.remove_nodes_from(to_remove)
            self.G.graph['ignoreInt'] = ignoreInt

        # 2. degree 0
        to_remove = [n for n in self.G.nodes() if self.G.degree(n) == 0]
        self.G.remove_nodes_from(to_remove)

        # 3.metadata
        #H = get_homophily(self.G)
        h = get_similitude(self.G)
        b = get_minority_fraction(self.G)
        k, km, kM = get_average_degrees(self.G)
        m = get_min_degree(self.G)
        density = get_density(self.G)

        self.G.graph["fullname"] = os.path.basename(datafn).replace(".gpickle","")
        self.G.graph['kind'] = self.kind
        self.G.graph['density'] = density

        for param in ['N','m','B','H','Hmm','HMM','i','x']:
            if param not in self.G.graph:
                self.G.graph[param] = get_param(datafn, param)

        # self.G.graph['N'] = get_param(datafn, "N") or self.G.number_of_nodes()
        # self.G.graph['m'] = get_param(datafn, "m") or m
        # self.G.graph['density'] = density
        # self.G.graph['B'] = get_param(datafn, "B") or b
        #
        # if 'H' not in self.G.graph:
        #     self.G.graph['H'] = get_param(datafn, "H") or H
        # if 'Hmm' not in self.G.graph:
        #     self.G.graph['Hmm'] = get_param(datafn, "Hmm")
        # if 'HMM' not in self.G.graph:
        # self.G.graph['HMM'] = get_param(datafn, "HMM")
        # self.G.graph['i'] = get_param(datafn, "i")
        # self.G.graph['x'] = get_param(datafn, "x")

        self.G.graph['n'] = self.G.number_of_nodes()
        self.G.graph['e'] = self.G.number_of_edges()
        self.G.graph['min_degree'] = m
        self.G.graph['h'] = h
        self.G.graph['b'] = b
        self.G.graph['k'] = k
        self.G.graph['km'] = km
        self.G.graph['kM'] = kM


    def info(self):
        '''
        Prints a summary of the network, including its attributes.
        '''
        print(nx.info(self.G))
        print(self.G.graph)

    def save(self, root):
        fn = os.path.join(root, self.G.graph['fullname'])
        write_gpickle(self.G, fn)