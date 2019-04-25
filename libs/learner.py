import warnings

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import os
import sys
from collections import Counter

import networkx as nx
import numpy as np
from sklearn.model_selection import train_test_split
from collections import defaultdict
from itertools import combinations

from libs.collective.collectivehandler import Collective
from libs.relational.relationalhandler import Relational
from libs.utils.loggerinit import *
import pickle
from heapq import *
from libs.sampling.node_ranking import NodeRanking
from libs.sampling.network_sample import NetworkSample
import pandas as pd
import operator

MINPNODES = 0.01
MAXPNODES = 0.9

SIZETEST = 100
MINPEDGES = 0.01
MAXPEDGES = 0.19


class Learner(object):

    def __init__(self, datafn, knownfn, pseeds, sampling, LC, RC, CI, label, RCattributes, LCattributes, test=False, ignore=None, seed=None):
        self.datafn = datafn
        self.knownfn = knownfn
        self.pseeds = pseeds
        self.sampling = sampling
        self.LC = LC
        self.RC = RC
        self.CI = CI
        self.label = label
        self.RCattributes = RCattributes
        self.LCattributes = LCattributes
        self.local = None
        self.relational = None
        self.collective = None
        self._data = None
        self.G = None
        self.Gseeds = None
        self.test = test
        self.ignore = ignore
        self.seed = seed
        np.random.seed(self.seed)
        self.ns = None
        self.nodeattributefn = None

    def initialize(self):
        self._load_full_graph()
        seednodes = self._load_seeds()
        self._training_sample(seednodes)
        self._load_data()
        self._validate_training_sample()
        self._init_modules()

    def set_nodes_attributes_fn(self, nodeattributefn):
        self.nodeattributefn = nodeattributefn

    def _get_subgraph(self,G,nnodes):
        nodes = G.nodes()
        np.random.shuffle(nodes)
        nodes = nodes[:nnodes]
        return G.subgraph(nodes).copy()

    def _load_graph_from_file(self):

        if os.path.exists(self.datafn) and self.datafn.endswith('.gpickle'):

            try:
                self.G = nx.read_gpickle(self.datafn)
            except:
                try:
                    with open(self.datafn, 'rb') as f:
                        self.G = pickle.load(f,encoding='latin1')
                except Exception as ex:
                    logging.error(ex)
                    logging.error('Could not open graph: {}'.format(self.datafn))
                    sys.exit(0)

        elif os.path.exists(self.datafn) and self.datafn.endswith('.edgelist'):

            if self.nodeattributefn is None:
                logging.error('*.edgelist file requires a *.csv file containing node attributes (argument: nafn).')
                sys.exit(0)

            df = pd.read_csv(self.nodeattributefn, index_col=False, header=None, names=['node','label'])

            self.G = nx.read_edgelist(self.datafn)
            self.G = nx.relabel_nodes(self.G, {n:int(n) for n in self.G.nodes()}, copy=True)

            node_attributes = {}
            for id,row in df.iterrows():
                node_attributes[row.node] = row.label
                if row.node not in self.G:
                    self.G.add_node(row.node)

            self.G.name = os.path.basename(self.datafn).replace('.edgelist','')
            self.G.graph['attributes'] = [self.label]
            self.G.graph['label'] = self.label

            nx.set_node_attributes(self.G, name=self.label, values=node_attributes)

        else:
            logging.error('{} does not exit or is not an NX network.'.format(self.datafn))
            sys.exit(0)

        return self.G

    def _load_full_graph(self):

        self._load_graph_from_file()

        # ignore from label
        if self.ignore is not None:
            toremove = [n[0] for n in self.G.nodes(data=True) if n[1][self.label] == self.ignore]
            self.G.remove_nodes_from(toremove)
            logging.info('{} nodes removed ({} == {})'.format(len(toremove), self.label, self.ignore))

        # ignore if most of attributes are unknown
        if self.ignore is not None:
            toremove = [n[0] for n in self.G.nodes(data=True) if sum([int(v != self.ignore) for v in n[1].values()]) < 2]
            self.G.remove_nodes_from(toremove)
            logging.info('{} nodes removed (most of their attributes are unknown - only 1 is known)'.format(len(toremove)))

        # small network when testing
        if self.test:
            self.G = self._get_subgraph(self.G, SIZETEST)

        # removing singletons
        toremove = [n for n in self.G.nodes() if self.G.degree(n) == 0]
        self.G.remove_nodes_from(toremove)
        logging.info('{} nodes removed (degree 0 - no neighbor information)'.format(len(toremove)))

        # There should at least be 3 elements in every class
        tmp = Counter([self.G.node[n][self.label] for n in self.G.nodes()])
        for k, v in tmp.items():
            if v <= 3:
                toremove = [n for n in self.G.nodes() if self.G.node[n][self.label] == k]
                self.G.remove_nodes_from(toremove)
                logging.info('{} nodes removed ({} label == {} element)'.format(len(toremove), k, v))

        self.G.name = self.datafn.split('/')[-1].split('.gpickle')[0]
        logging.info('Data Network:\n{}'.format(nx.info(self.G)))

    def _load_seeds(self):
        seednodes = None

        if self.sampling == 'nodes':
            seednodes = self._sampling_random_nodes()
        elif self.sampling == 'biasednodes':
            seednodes = self._sampling_random_biased_nodes()


        elif self.sampling == 'maxdispersion':
            seednodes = self._sampling_random_max_dispersion_1hop()
        elif self.sampling == 'maxdispersionedge':
            seednodes = self._sampling_random_max_dispersion_edge()
        elif self.sampling == 'maxdispersionedgeunique':
            seednodes = self._sampling_random_max_dispersion_edge_unique()


        elif self.sampling == 'maxdispersion1random':
            seednodes = self._sampling_random_max_dispersion_g1(True)
        elif self.sampling == 'maxdispersion1nonrandom':
            seednodes = self._sampling_random_max_dispersion_g1(False)

        elif self.sampling == 'edgedispersion':
            seednodes = self._sampling_random_edge_dispersion_g1()
        elif self.sampling == 'edgemaxdispersion':
            seednodes = self._sampling_random_edges_max_dispersion()


        elif self.sampling == 'edge2maxdispersion':
            seednodes = self._sampling_random_2edges_max_dispersion()



        elif self.sampling == 'mindegree10':
            seednodes = self._sampling_random_biased_nodes_degree(10)


        elif self.sampling == 'edges':
            seednodes = self._sampling_random_edges()

        elif self.sampling == 'nedges':
            seednodes = self._sampling_random_nedges()

        elif self.sampling == 'snowball':
            seednodes  = self._sampling_snowball()

        elif self.sampling == 'degreeASC':
            seednodes  = self._sampling_degree(True)
        elif self.sampling == 'degreeDESC':
            seednodes  = self._sampling_degree(False)
        elif self.sampling == 'degreeMIX':
            seednodes  = self._sampling_degree(None)


        elif self.sampling == 'trianglesASC':
            seednodes  = self._sampling_triangles(True)
        elif self.sampling == 'trianglesDESC':
            seednodes  = self._sampling_triangles(False)
        elif self.sampling == 'trianglesMIX':
            seednodes = self._sampling_triangles(None)

        elif self.sampling == 'pagerankASC':
            seednodes  = self._sampling_page_rank(True)
        elif self.sampling == 'pagerankDESC':
            seednodes  = self._sampling_page_rank(False)

        elif self.sampling in ['percolationASC','percolationDESC'] and self.knownfn is not None:
            seednodes = self._sampling_percolation()

        elif self.sampling == 'friendshipParadox1':
            seednodes = self._sampling_friendship_paradox(1)
        elif self.sampling == 'friendshipParadox2':
            seednodes = self._sampling_friendship_paradox(2)
        elif self.sampling == 'friendshipParadox3':
            seednodes = self._sampling_friendship_paradox(3)
        elif self.sampling == 'friendshipParadox4':
            seednodes = self._sampling_friendship_paradox(4)
        elif self.sampling == 'friendshipParadox5':
            seednodes = self._sampling_friendship_paradox(5)
        elif self.sampling == 'wedges':
            seednodes = self._sampling_friendship_paradox(6)
        elif self.sampling == 'friendshipParadox7':
            seednodes = self._sampling_friendship_paradox(7)
        elif self.sampling == 'friendshipParadox8':
            seednodes = self._sampling_friendship_paradox(8)
        elif self.sampling == 'friendshipParadox9':
            seednodes = self._sampling_friendship_paradox(9)
        elif self.sampling == 'friendshipParadox10':
            seednodes = self._sampling_friendship_paradox(10)
        elif self.sampling == 'friendshipParadox11':
            seednodes = self._sampling_friendship_paradox(11)
        elif self.sampling == 'friendshipParadox12':
            seednodes = self._sampling_friendship_paradox(12)
        elif self.sampling == 'wedges3a':
            seednodes = self._sampling_friendship_paradox(13)
        elif self.sampling == 'wedges3b':
            seednodes = self._sampling_friendship_paradox(14)
        elif self.sampling == 'friendshipParadoxSeeds':
            seednodes = self._sampling_friendship_paradox(15)

        elif self.sampling == 'egoNetwork':
            seednodes = self._sampling_ego_network('ego')
        elif self.sampling == '1Hop':
            seednodes = self._sampling_ego_network('1hop')
        elif self.sampling == '1HopEven':
            seednodes = self._sampling_ego_network('1hopeven')
        elif self.sampling == '1HopOdd':
            seednodes = self._sampling_ego_network('1hopodd')


        elif self.sampling == 'randomWalk':
            seednodes = self._sampling_random_walk('original')
        elif self.sampling == 'randomWalkUnique':
            seednodes = self._sampling_random_walk('unique')
        elif self.sampling == 'randomWalkEven':
            seednodes = self._sampling_random_walk('even')
        elif self.sampling == 'randomWalkOdd':
            seednodes = self._sampling_random_walk('odd')


        elif self.knownfn is None:
            logging.info('{} sampling is not supported.'.format(self.sampling))
            sys.exit(0)

        if seednodes is not None:
            logging.info('{} ({}%) seed nodes using {} sampling'.format(len(seednodes),round(len(seednodes)*100/float(self.G.number_of_nodes())),self.sampling))

        return seednodes

    ############################################################
    # SAMPLING METHODS START
    ############################################################

    def _sampling_random_edges(self):

        if (self.pseeds >= MINPNODES and self.pseeds <= MAXPNODES):

            nnodes = int(round(self.pseeds * self.G.number_of_nodes()))
            logging.info('sampling by edges: {} ({}%) nodes'.format(nnodes,self.pseeds))
            seededges = []
            seednodes = set()

            try:
                edges = list(self.G.edges())
                edgeids = np.arange(self.G.number_of_edges()).tolist()
                np.random.shuffle(edgeids)

                while len(seednodes) < nnodes:
                    id = edgeids.pop(0)
                    seededges.append(edges[id])
                    seednodes.add(edges[id][0])
                    seednodes.add(edges[id][1])

            except Exception as ex:
                seededges = None
                logging.warning(ex)
                pass

            if seededges is None:
                logging.error('There is not good known sample.')
                sys.exit(0)

            if len(seednodes) == self.G.number_of_nodes():
                logging.error('Training sample cannot be the whole graph. bye!')
                sys.exit(0)

            self.Gseeds = self.G.edge_subgraph(seededges)
            return list(seednodes)

        else:
            logging.error('{}: {}% of edges is not a valid number. Try >={} and <={}'.format(self.sampling, self.pseeds, MINPEDGES, MAXPEDGES))

    def _sampling_random_nedges(self):

        if (self.pseeds >= MINPNODES and self.pseeds <= MAXPNODES):

            nnodes = int(round(self.pseeds * self.G.number_of_nodes()))
            logging.info('sampling nedges: {} ({}%) nodes'.format(nnodes,self.pseeds*100))
            seededges = None

            try:
                edges = self.G.edges()
                np.random.shuffle(edges)

                seededges = set()
                nodes = set()

                while len(nodes) < nnodes:
                    id = np.random.choice(range(len(edges)), 1, replace=False)[0]
                    seededges.add(id)
                    nodes.add(edges[id][0])
                    nodes.add(edges[id][1])

                seededges = [edges[ei] for ei in seededges]
            except:
                pass

            if seededges is None:
                logging.error('There is not good known sample.')
                sys.exit(0)

            tmp = nx.Graph(list(seededges))
            seednodes = tmp.nodes()

            if len(seednodes) == self.G.number_of_nodes():
                logging.error('Training sample cannot be the whole graph. bye!')
                sys.exit(0)

            logging.debug('Nodes from graph and from sampling: {} = {} = {} ?'.format(len(nodes),len(seednodes),len(set(nodes).intersection(set(seednodes)))))
            return seednodes

        else:
            logging.error('{}: {}% of edges is not a valid number. Try >={} and <={}'.format(self.sampling, self.pseeds, MINPNODES, MAXPNODES))

    def _sampling_random_nodes(self):

        if (self.pseeds >= MINPNODES and self.pseeds <= MAXPNODES):
            nodes = self.G.nodes()
            nnodes = int(round(self.pseeds * self.G.number_of_nodes()))
            logging.info('sampling nodes: {} ({}%) nodes'.format(nnodes,self.pseeds))
            seednodes = None

            try:
                seednodes = np.random.choice(nodes, nnodes, replace=False)
            except:
                pass

            if seednodes is None:
                logging.error('There is not good known sample.')
                sys.exit(0)

            if len(seednodes) == self.G.number_of_nodes():
                logging.error('Training sample cannot be the whole graph. bye!')
                sys.exit(0)

            return seednodes

        else:
            logging.error('{}: {}% of nodes is not a valid number. Try >={} and <={}'.format(self.sampling, self.pseeds, MINPNODES, MAXPNODES))
            sys.exit(0)

    def _sampling_random_biased_nodes(self):

        maxtries = self.G.number_of_nodes() * 10
        global_precision = 1 #3

        if (self.pseeds >= MINPNODES and self.pseeds <= MAXPNODES):
            nodes = list(self.G.nodes())
            np.random.shuffle(nodes)

            nnodes = int(round(self.pseeds * self.G.number_of_nodes()))
            logging.info('sampling biased nodes: {} ({}%) nodes'.format(nnodes, self.pseeds))

            labels = list(nx.get_node_attributes(self.G, self.label).values())
            Bglobal = round(Counter(labels).most_common(1)[0][1] / self.G.number_of_nodes(),global_precision)
            Hglobal = round(sum([int(self.G.node[edge[0]][self.label] == self.G.node[edge[1]][self.label]) for edge in self.G.edges()])/self.G.number_of_edges(),global_precision)

            labels = list(set(labels))
            H = Counter(['{}'.format(sorted([labels.index(self.G.node[edge[0]][self.label]), labels.index(self.G.node[edge[1]][self.label])])) for edge in self.G.edges()])
            H00 = round(H['[0, 0]'] / self.G.number_of_edges(),global_precision)
            H11 = round(H['[1, 1]'] / self.G.number_of_edges(),global_precision)
            H01 = round(H['[0, 1]'] / self.G.number_of_edges(),global_precision)
            logging.info('Bglobal:{} | Hglobal:{}, H00:{}, H11:{}: H01:{}'.format(Bglobal, Hglobal, H00, H11, H01))

            tries = 0
            local_precision = 2

            logging.info('Inferring sample...')
            while True:

                if tries >= maxtries:
                    logging.error('{} tries. There is not good biased sample.'.format(maxtries))
                    sys.exit(0)

                try:
                    seednodes = np.random.choice(nodes, nnodes, replace=False)

                    gseeds = self.G.subgraph(seednodes)

                    if gseeds.number_of_edges() < 3: #5
                        tries += 1
                        continue

                    # number of
                    h00 = [int(self.G.node[edge[0]][self.label] == self.G.node[edge[1]][self.label]) for edge in gseeds.edges() if self.G.node[edge[0]][self.label] == labels[0]]
                    h11 = [int(self.G.node[edge[0]][self.label] == self.G.node[edge[1]][self.label]) for edge in gseeds.edges() if self.G.node[edge[0]][self.label] == labels[1]]
                    h01 = [int(self.G.node[edge[0]][self.label] != self.G.node[edge[1]][self.label]) for edge in gseeds.edges()]

                    # print('{},{},{}'.format(sum(h00),sum(h11),gseeds.number_of_edges()))

                    # percentages
                    B = [round(v/gseeds.number_of_nodes(),local_precision) for v in Counter(list(nx.get_node_attributes(gseeds, self.label).values())).values()]
                    h00 = round(sum(h00) / float(gseeds.number_of_edges()),local_precision)
                    h11 = round(sum(h11) / float(gseeds.number_of_edges()),local_precision)
                    h01 = round(sum(h01) / float(gseeds.number_of_edges()), local_precision)
                    h = round(sum([int(gseeds.node[edge[0]][self.label] == gseeds.node[edge[1]][self.label]) for edge in gseeds.edges()])/gseeds.number_of_edges(),local_precision)


                    # logging.info('B:{}, diff:{} || Hglobal:{}, h:{} ({})|| H00:{}, h00:{} || H11:{}, h11:{} || H01:{}, h01:{} || DHoHe:{}, Dhohe:{} ...'.format(B, abs(B[0] - B[1]),
                    #                                                                                                                                                        Hglobal, h, abs(h - Hglobal),
                    #                                                                                                                                                        H00, h00,
                    #                                                                                                                                                        H11, h11,
                    #                                                                                                                                                        H01, h01,
                    #                                                                                                                                                        abs(H00 - H11),
                    #                                                                                                                                                        abs(h00 - h11)))

                    thres = 0.01
                    if Hglobal >= 0.5:
                        if abs(h - Hglobal) < thres \
                            and abs(h00 - h11) < thres \
                            and abs(B[0] - B[1]) < thres \
                            and abs(abs(h00 - h11)-abs(H00 - H11)) < thres \
                            and (abs(B[0] - Bglobal) < thres or abs(B[1] - Bglobal) < thres):
                            logging.info('B:{}, diff:{} || Hglobal:{}, h:{} ({})|| H00:{}, h00:{} || H11:{}, h11:{} || H01:{}, h01:{} || DHoHe:{}, Dhohe:{} || (GOOD-Hete)'.format(B, abs(B[0] - B[1]),
                                                                                                                                                                                   Hglobal, h,
                                                                                                                                                                                   abs(h - Hglobal),
                                                                                                                                                                                   H00, h00,
                                                                                                                                                                                   H11, h11,
                                                                                                                                                                                   H01, h01,
                                                                                                                                                                                   abs(H00 - H11),
                                                                                                                                                                                   abs(h00 - h11)))
                            logging.info('{} tries'.format(tries))
                            break
                        else:
                            tries += 1
                    else:
                        if abs(h - Hglobal) < thres \
                            and abs(h01 - H01) < thres \
                            and abs(B[0] - B[1]) < thres \
                            and (abs(B[0] - Bglobal) < thres or abs(B[1] - Bglobal) < thres):
                            logging.info('B:{}, diff:{} || Hglobal:{}, h:{} ({})|| H00:{}, h00:{} || H11:{}, h11:{} || H01:{}, h01:{} || DHoHe:{}, Dhohe:{} || (GOOD-Hete)'.format(B, abs(B[0] - B[1]),
                                                                                                                                                                                   Hglobal, h,
                                                                                                                                                                                   abs(h - Hglobal),
                                                                                                                                                                                   H00, h00,
                                                                                                                                                                                   H11, h11,
                                                                                                                                                                                   H01, h01,
                                                                                                                                                                                   abs(H00 - H11),
                                                                                                                                                                                   abs(h00 - h11)))
                            logging.info('{} tries'.format(tries))
                            break
                        else:
                            tries += 1

                except Exception as ex:
                    logging.info(ex)
                    tries += 1
                    pass

            if seednodes is None:
                logging.error('There is not good known sample.')
                sys.exit(0)

            if len(seednodes) == self.G.number_of_nodes():
                logging.error('Training sample cannot be the whole graph. bye!')
                sys.exit(0)

            return seednodes

        else:
            logging.error('{}: {}% of nodes is not a valid number. Try >={} and <={}'.format(self.sampling, self.pseeds, MINPNODES, MAXPNODES))



    def _sampling_random_max_dispersion_1hop(self):

        if (self.pseeds >= MINPNODES and self.pseeds <= MAXPNODES):
            nodes = list(self.G.nodes())
            nnodes = int(round(self.pseeds * self.G.number_of_nodes()))
            logging.info('sampling node by {}: {} ({}%) nodes'.format(self.sampling,nnodes,self.pseeds))
            seednodes = set()

            np.random.shuffle(nodes)
            tries = len(nodes)


            while len(seednodes) < nnodes and tries > 0 and len(nodes)>0:
                tries -= 1

                try:
                    vi = nodes.pop(0)
                    values = {vj:nx.dispersion(self.G, vi, vj) for vj in nx.neighbors(self.G, vi)}

                    if len(values) == 0:
                        continue

                    values = sorted(values.items(), key=operator.itemgetter(1), reverse=True)
                    vj = values[0][0]

                    if vj in seednodes:
                        np.random.shuffle(nodes)
                        # vi = nodes.pop(0)
                    else:
                        seednodes.add(vj)
                        # vi = vj

                except:
                    pass

            if seednodes is None:
                logging.error('There is not good known sample.')
                sys.exit(0)

            seednodes = list(seednodes)

            if len(seednodes) == self.G.number_of_nodes():
                logging.error('Training sample cannot be the whole graph. bye!')
                sys.exit(0)

            return seednodes

        else:
            logging.error('{}: {}% of nodes is not a valid number. Try >={} and <={}'.format(self.sampling, self.pseeds, MINPNODES, MAXPNODES))
            sys.exit(0)

    def _sampling_random_max_dispersion_edge(self):

        if (self.pseeds >= MINPNODES and self.pseeds <= MAXPNODES):
            nodes = list(self.G.nodes())
            nnodes = int(round(self.pseeds * self.G.number_of_nodes()))
            logging.info('sampling node by {}: {} ({}%) nodes'.format(self.sampling,nnodes,self.pseeds))
            seednodes = set()

            np.random.shuffle(nodes)
            tries = len(nodes)


            while len(seednodes) < nnodes and tries > 0 and len(nodes)>0:
                tries -= 1

                try:
                    vi = nodes.pop(0)
                    values = {vj:nx.dispersion(self.G, vi, vj) for vj in nx.neighbors(self.G, vi)}

                    if len(values) == 0:
                        continue

                    values = sorted(values.items(), key=operator.itemgetter(1), reverse=True)
                    vj = values[0][0]

                    if vj in seednodes:
                        np.random.shuffle(nodes)
                        # vi = nodes.pop(0)
                    else:
                        seednodes.add(vi)
                        seednodes.add(vj)
                        # vi = vj

                except:
                    pass

            if seednodes is None:
                logging.error('There is not good known sample.')
                sys.exit(0)

            seednodes = list(seednodes)

            if len(seednodes) == self.G.number_of_nodes():
                logging.error('Training sample cannot be the whole graph. bye!')
                sys.exit(0)

            return seednodes

        else:
            logging.error('{}: {}% of nodes is not a valid number. Try >={} and <={}'.format(self.sampling, self.pseeds, MINPNODES, MAXPNODES))
            sys.exit(0)

    def _sampling_random_max_dispersion_edge_unique(self):

        if (self.pseeds >= MINPNODES and self.pseeds <= MAXPNODES):
            nodes = list(self.G.nodes())
            nnodes = int(round(self.pseeds * self.G.number_of_nodes()))
            logging.info('sampling node by {}: {} ({}%) nodes'.format(self.sampling,nnodes,self.pseeds))
            seednodes = set()

            np.random.shuffle(nodes)
            tries = len(nodes)


            while len(seednodes) < nnodes and tries > 0 and len(nodes)>0:
                tries -= 1

                try:
                    vi = nodes.pop(0)
                    values = {vj:nx.dispersion(self.G, vi, vj) for vj in nx.neighbors(self.G, vi)}

                    if len(values) == 0:
                        continue

                    values = sorted(values.items(), key=operator.itemgetter(1), reverse=True)
                    vj = values[0][0]

                    if vi not in seednodes or vj not in seednodes:
                        seednodes.add(vi)
                        seednodes.add(vj)
                        # vi = vj
                    else:
                        np.random.shuffle(nodes)
                        # vi = nodes.pop(0)

                except:
                    pass

            if seednodes is None:
                logging.error('There is not good known sample.')
                sys.exit(0)

            seednodes = list(seednodes)

            if len(seednodes) == self.G.number_of_nodes():
                logging.error('Training sample cannot be the whole graph. bye!')
                sys.exit(0)

            return seednodes

        else:
            logging.error('{}: {}% of nodes is not a valid number. Try >={} and <={}'.format(self.sampling, self.pseeds, MINPNODES, MAXPNODES))
            sys.exit(0)





    def _sampling_random_max_dispersion_g1(self, random=True):

        if (self.pseeds >= MINPNODES and self.pseeds <= MAXPNODES):
            nodes = list(self.G.nodes())
            nnodes = int(round(self.pseeds * self.G.number_of_nodes()))
            logging.info('sampling node by maxdispersion1HOP: {} ({}%) nodes'.format(nnodes,self.pseeds))
            seednodes = set()

            vj = None
            tries = self.G.number_of_nodes()
            while len(seednodes) < nnodes and len(nodes) > 0 and tries > 0:

                tries -= 1

                try:

                    np.random.shuffle(nodes)
                    if random or vj is None:
                        vi = nodes.pop(0)
                    else:
                        vi = vj

                    values = {vj:nx.dispersion(self.G, vi, vj) for vj in nx.neighbors(self.G, vi)}

                    if len(values) == 0:
                        vj = None
                        continue

                    values = sorted(values.items(), key=operator.itemgetter(1), reverse=True)
                    vj = values[0][0]

                    if values[0][1] < 1:
                        continue

                    if vi not in seednodes or vj not in seednodes:
                        logging.info('dispersion vi:{}, vj:{} = {}'.format(vi, vj, values[0][1]))
                        seednodes.add(vi)
                        seednodes.add(vj)



                except:
                    pass

            if seednodes is None:
                logging.error('There is not good known sample.')
                sys.exit(0)

            seednodes = list(seednodes)

            if len(seednodes) == self.G.number_of_nodes():
                logging.error('Training sample cannot be the whole graph. bye!')
                sys.exit(0)

            return seednodes

        else:
            logging.error('{}: {}% of nodes is not a valid number. Try >={} and <={}'.format(self.sampling, self.pseeds, MINPNODES, MAXPNODES))
            sys.exit(0)



    # biased b,h + dispersion

    def _sampling_random_edge_dispersion_g1(self):

        if (self.pseeds >= MINPNODES and self.pseeds <= MAXPNODES):
            edges = list(self.G.edges())
            edgesid = np.arange(len(edges))

            nnodes = int(round(self.pseeds * self.G.number_of_nodes()))
            logging.info('sampling node by {}: {} ({}%) nodes'.format(self.sampling,nnodes,self.pseeds))
            seednodes = set()


            while len(seednodes) < nnodes and len(edgesid) > 0:
                np.random.shuffle(edgesid)

                try:
                    edgeid = edgesid.pop(0)
                    edge = edges[edgeid]

                    vi = edge[0]
                    vj = edge[1]

                    if nx.dispersion(self.G, vi, vj) > 1:
                        seednodes.add(vi)
                        seednodes.add(vj)
                except:
                    pass

            if seednodes is None:
                logging.error('There is not good known sample.')
                sys.exit(0)

            seednodes = list(seednodes)

            if len(seednodes) == self.G.number_of_nodes():
                logging.error('Training sample cannot be the whole graph. bye!')
                sys.exit(0)

            return seednodes

        else:
            logging.error('{}: {}% of nodes is not a valid number. Try >={} and <={}'.format(self.sampling, self.pseeds, MINPNODES, MAXPNODES))
            sys.exit(0)

    def _sampling_random_edges_max_dispersion(self):

        if (self.pseeds >= MINPNODES and self.pseeds <= MAXPNODES):
            nodes = list(self.G.nodes())
            edges = list(self.G.edges())
            edgesid = np.arange(len(edges))

            nnodes = int(round(self.pseeds * self.G.number_of_nodes()))
            logging.info('sampling node by {}: {} ({}%) nodes'.format(self.sampling,nnodes,self.pseeds))
            seednodes = set()

            while len(seednodes) < nnodes and len(edgesid) > 0:
                np.random.shuffle(edgesid)

                try:
                    edgeid = edgesid.pop(0)
                    edge = edges[edgeid]

                    vi = edge[0]
                    vj = edge[1]

                    seednodes.add(vi)
                    seednodes.add(vj)

                    if len(seednodes) > 0:
                        values = defaultdict(lambda :0)
                        for ni,n1 in enumerate(seednodes):
                            for n2 in seednodes[ni+1:]:
                                d = nx.dispersion(self.G, n1, n2)
                                values[n1] = max(values[n1],d)
                                values[n2] = max(values[n1],d)

                        for n,d in values.items():
                            if d == 0:
                                seednodes.remove(n)

                except:
                    pass

            if seednodes is None:
                logging.error('There is not good known sample.')
                sys.exit(0)

            seednodes = list(seednodes)

            if len(seednodes) == self.G.number_of_nodes():
                logging.error('Training sample cannot be the whole graph. bye!')
                sys.exit(0)

            return seednodes

        else:
            logging.error('{}: {}% of nodes is not a valid number. Try >={} and <={}'.format(self.sampling, self.pseeds, MINPNODES, MAXPNODES))
            sys.exit(0)



    def _sampling_random_2edges_max_dispersion(self):

        if (self.pseeds >= MINPNODES and self.pseeds <= MAXPNODES):
            nodes = list(self.G.nodes())
            edges = list(self.G.edges())
            edgesid = np.arange(len(edges))
            np.random.shuffle(edgesid)

            nnodes = int(round(self.pseeds * self.G.number_of_nodes()))
            logging.info('sampling node by {}: {} ({}%) nodes'.format(self.sampling,nnodes,self.pseeds))
            seednodes = set()
            tries = int(round(len(edges) * (len(edges)/2.)))
            taken = set()

            while len(seednodes) < nnodes and tries > 0 and len(edgesid) > 0:

                tries -= 1

                try:

                    i1,i2 = np.random.choice(edgesid,2,replace=False).tolist()

                    if i1 in taken and i2 in taken:
                        continue

                    edge1 = edges[i1]
                    edge2 = edges[i2]

                    vi = edge1[0]
                    vj = edge1[1]
                    d1 = nx.dispersion(self.G, vi, vj)

                    wi = edge2[0]
                    wj = edge2[1]
                    d2 = nx.dispersion(self.G, wi, wj)

                    if d1 > d2:
                        seednodes.add(vi)
                        seednodes.add(vj)
                        taken.add(i1)
                    else:
                        seednodes.add(wi)
                        seednodes.add(wj)
                        taken.add(i2)

                except:
                    pass

            if seednodes is None:
                logging.error('There is not good known sample.')
                sys.exit(0)

            seednodes = list(seednodes)

            if len(seednodes) == self.G.number_of_nodes():
                logging.error('Training sample cannot be the whole graph. bye!')
                sys.exit(0)

            return seednodes

        else:
            logging.error('{}: {}% of nodes is not a valid number. Try >={} and <={}'.format(self.sampling, self.pseeds, MINPNODES, MAXPNODES))
            sys.exit(0)




    def _sampling_random_biased_nodes_degree(self, minimun):
        if (self.pseeds >= MINPNODES and self.pseeds <= MAXPNODES):
            nodes = list(self.G.nodes())
            nnodes = int(round(self.pseeds * self.G.number_of_nodes()))
            logging.info('sampling nodes min degree 10: {} ({}%) nodes'.format(nnodes,self.pseeds))
            seednodes = set()

            try:
                np.random.shuffle(nodes)
                while len(seednodes) < nnodes:
                    node = nodes.pop(0)
                    if self.G.degree(node) >= minimun:
                        seednodes.add(node)
                seednodes = list(seednodes)
            except:
                pass

            if seednodes is None:
                logging.error('There is not good known sample.')
                sys.exit(0)

            if len(seednodes) == self.G.number_of_nodes():
                logging.error('Training sample cannot be the whole graph. bye!')
                sys.exit(0)

            return seednodes

        else:
            logging.error('{}: {}% of nodes is not a valid number. Try >={} and <={}'.format(self.sampling, self.pseeds, MINPNODES, MAXPNODES))



    def _sampling_page_rank(self, asc=True):
        if (self.pseeds >= MINPNODES and self.pseeds <= MAXPNODES):
            ns = NodeRanking.get_instance(NodeRanking.PAGERANK, self.G, self.pseeds)
            ns.compute_node_scores()
            seednodes = ns.rank_nodes(asc)
            return seednodes
        else:
            logging.error('{}: {}% of nodes is not a valid number. Try >={} and <={}'.format(self.sampling, self.sampling, self.pseeds, MINPNODES, MAXPNODES))

    def _sampling_degree(self, asc=True):
        if (self.pseeds >= MINPNODES and self.pseeds <= MAXPNODES):
            ns = NodeRanking.get_instance(NodeRanking.DEGREE, self.G, self.pseeds)
            ns.compute_node_scores()
            seednodes = ns.rank_nodes(asc)
            return seednodes
        else:
            logging.error('{}: {}% of nodes is not a valid number. Try >={} and <={}'.format(self.sampling, self.pseeds, MINPNODES, MAXPNODES))

    def _sampling_triangles(self, asc=True):
        if (self.pseeds >= MINPNODES and self.pseeds <= MAXPNODES):
            ns = NodeRanking.get_instance(NodeRanking.TRIANGLES, self.G, self.pseeds)
            ns.compute_node_scores()
            seednodes = ns.rank_nodes(asc)
            return seednodes
        else:
            logging.error('{}: {}% of nodes is not a valid number. Try >={} and <={}'.format(self.sampling, self.pseeds, MINPNODES, MAXPNODES))

    def _sampling_snowball(self):
        if (self.pseeds >= MINPNODES and self.pseeds <= MAXPNODES):
            ns = NetworkSample.get_instance(NetworkSample.SNOWBALL, self.G, self.pseeds)
            ns.compute_sample()
            return ns.get_seednodes()
        else:
            logging.error('{}: {}% of nodes is not a valid number. Try >={} and <={}'.format(self.sampling, self.pseeds, MINPNODES, MAXPNODES))

    def _sampling_percolation(self):
        nodes = []
        if self.knownfn.endswith('.txt') and os.path.exists(self.knownfn):
            with open(self.knownfn,'rb') as f:
                nodes = f.readlines()
            nodes = [int(x.strip()) for x in nodes]
        nnodes = int(round(self.G.number_of_nodes() * self.pseeds))
        return nodes[:nnodes]

    def _sampling_friendship_paradox(self, version):
        if (self.pseeds >= MINPNODES and self.pseeds <= MAXPNODES):
            ns = NetworkSample.get_instance(NetworkSample.FRIENDSHIPPARADOX, self.G, self.pseeds)
            ns.compute_sample(version)
            self.Gseeds = ns.Gsample
            return ns.get_seednodes()
        else:
            logging.error('{}: {}% of nodes is not a valid number. Try >={} and <={}'.format(self.sampling, self.pseeds, MINPNODES, MAXPNODES))

    def _sampling_ego_network(self, version):
        if self.pseeds >= MINPNODES and self.pseeds <= MAXPNODES:
            ns = NetworkSample.get_instance(NetworkSample.EGO, self.G, self.pseeds)
            ns.compute_sample(version)
            return ns.get_seednodes()
        else:
            logging.error('{}: {}% of nodes is not a valid number. Try >={} and <={}'.format(self.sampling, self.pseeds, MINPNODES, MAXPNODES))

    def _sampling_random_walk(self, version):
        if self.pseeds >= MINPNODES and self.pseeds <= MAXPNODES:
            ns = NetworkSample.get_instance(NetworkSample.RANDOMWALK, self.G, self.pseeds)
            ns.compute_sample(version)
            self.Gseeds = ns.Gsample
            return ns.get_seednodes()
        else:
            logging.error('{}: {}% of nodes is not a valid number. Try >={} and <={}'.format(self.sampling, self.pseeds, MINPNODES, MAXPNODES))

    ############################################################
    # SAMPLING METHODS END
    ############################################################

    def _training_sample(self, seednodes=None):
        print('Training sample')
        try:

            if self.Gseeds is None:
                nodes = None

                if self.knownfn is None and seednodes is not None:
                    pass
                elif self.knownfn is None and seednodes is None:
                    logging.error('There is no training data.')
                    sys.exit(0)
                elif self.knownfn.endswith('.gpickle') and os.path.exists(self.knownfn):
                    nodes = nx.read_gpickle(self.knownfn).nodes()
                elif self.knownfn.endswith('.pickle') and os.path.exists(self.knownfn):
                    with open(self.knownfn,'rb') as f:
                        nodes = pickle.load(f)
                elif self.knownfn.endswith('.txt') and seednodes is not None:
                    pass

                else:
                    logging.error('Known data ({}) does not exist.'.format(self.knownfn))
                    sys.exit(0)


                if nodes is not None:
                    if (self.pseeds > 0 and self.pseeds <= 1.0):
                        # nnodes = int(round(self.pseeds * self.G.number_of_nodes()))
                        nnodes = int(round(self.pseeds * len(nodes)))
                        seednodes = np.random.choice(nodes, nnodes, replace=False)
                    else:
                        logging.error('Wrong percentage of seeds ({} given).'.format(self.pseeds))
                        sys.exit(0)

                if seednodes is None:
                    logging.error('There is not good known sample.')
                    sys.exit(0)

                if len(seednodes) >= self.G.number_of_nodes():
                    logging.error('Training sample is too big (or bigger or equal than data)')
                    sys.exit(0)

                if self.knownfn is None and seednodes is not None:
                    tries = 100
                    while tries > 0:
                        tmp = self.G.subgraph(seednodes).copy()
                        if tmp.number_of_edges() < 3:
                            tries -= 1
                            logging.info('There are no enough edges in this subgraph (Gseeds - {}): {} edges (try #{}).'.format(self.sampling, tmp.number_of_edges(),100-tries))
                            seednodes = self._load_seeds()
                        else:
                            break

                self.Gseeds = self.G.subgraph(seednodes).copy()

            else:
                logging.info('The subgraph (Gseeds) is given. name {}, attributes {}, {} edges, and {} nodes.'.format(self.Gseeds.name, self.Gseeds.graph['attributes'], self.Gseeds.number_of_edges(), self.Gseeds.number_of_nodes()))

            if self.Gseeds.number_of_edges() < 3:
                logging.info('There are no enough edges in this subgraph (Gseeds): {}.'.format(self.Gseeds.number_of_edges()))
                sys.exit(0)

            logging.info('{} nodes as seeds ({}% using {} sampling)'.format(len(seednodes), self.pseeds * 100, self.sampling))
            logging.info('SEEDS | head:{} | tail:{}'.format(seednodes[:5],seednodes[-5:]))
            logging.info('Gseeds: \n{}'.format(nx.info(self.Gseeds)))

        except Exception as ex:
            logging.warning(ex)
            logging.warning('Seeds are not a fraction of the graph')

            if self.knownfn is not None and os.path.exists(self.knownfn) and self.knownfn.endswith('.gpickle'):
                self.Gseeds = nx.read_gpickle(self.knownfn)
                logging.info('Gseeds loaded: {}'.format(self.knownfn))
                logging.info(nx.info(self.Gseeds))
            else:
                logging.error('Seeds are not given as a separate graph')
                sys.exit(0)
            pass

    def _load_data(self):
        labels = list(sorted(set([self.Gseeds.node[n][self.label] for n in self.Gseeds.nodes()])))
        self.Gseeds.graph['labels'] = labels
        counter = 0

        while len(labels) == 1:
            logging.warning('The set of seed nodes include only 1 class (unbalanced)')

            self.Gseeds = None
            seednodes = self._load_seeds()
            self._training_sample(seednodes)
            labels = list(sorted(set([self.Gseeds.node[n][self.label] for n in self.Gseeds.nodes()])))
            self.Gseeds.graph['labels'] = labels

            counter+=1
            if counter == 100:
                logging.warning('The set of seed nodes include only 1 class (unbalanced, we tried 100 times)')
                sys.exit(0)

        logging.info('{} Classes.'.format(len(labels)))
        logging.info(labels)

    def _validate_training_sample(self):
        # training sample that
        toremove = [n[0] for n in self.G.nodes(data=True) if n[0] not in self.Gseeds and n[1][self.label] not in self.Gseeds.graph['labels']]
        self.G.remove_nodes_from(toremove)
        logging.info('{} nodes removed (training data with unknown prior - no label in seeds)'.format(len(toremove)))

    def _init_modules(self):
        self.relational = Relational.get_instance(self.LC, self.RC, self.G, self.Gseeds, self.label, self.RCattributes, self.LCattributes, self.ignore)
        self.collective = Collective.get_instance(self.CI, self.relational, self.test)

    def learn(self):
        logging.info('Learning phase ({} + {})'.format(self.LC, self.RC))
        self.collective.learn()

    def classify(self):
        logging.info('Classification phase ({} + {})'.format(self.RC, self.CI))
        self.collective.classify()

    def evaluation(self):
        return self.collective.evaluation()