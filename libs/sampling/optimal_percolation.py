from libs.sampling.node_ranking import NodeRanking
import networkx as nx
from joblib import Parallel, delayed
import multiprocessing
import numpy as np
from heapq import *
from libs.utils.loggerinit import *

class OptimalPercolation(NodeRanking):
    BIGNET = 1000
    MAXDIAMETER = 10
    PNODES = None #0.05

    def __init__(self, G, pseeds):
        super(OptimalPercolation, self).__init__(max(nx.connected_component_subgraphs(G), key=len), pseeds)

    def compute_node_scores(self):
        self._define_cutoff()
        scores_ball = self._percolation()
        self.scores = self._best_percolation(scores_ball)

    def _percolation(self):
        logging.info('Computing diameter...')
        diameter = self.MAXDIAMETER if self.G.number_of_nodes() > self.BIGNET else nx.diameter(self.G)
        logging.info('Diameter: {}'.format(diameter))
        ncores = multiprocessing.cpu_count()
        njobs = diameter if diameter < ncores else ncores
        results = Parallel(n_jobs=njobs)(delayed(self._percolation_ball_parallel)(l) for l in np.arange(1,diameter))
        return results

    def _define_cutoff(self):
        N = self.G.number_of_nodes()
        if N >= self.BIGNET:
            self.CUTOFF = 0.01
        else:
            self.CUTOFF = 1/ N

    def _percolation_ball_parallel(self,l):
        scores = []
        graph = self.G.copy()
        N = self.G.number_of_nodes()
        logging.info('=== Ball l:{} ==='.format(l))
        while True:
            ntoremove = 1 if self.PNODES is None else int(round(self.PNODES * graph.number_of_nodes()))
            CIs = [(OptimalPercolation.get_ci(graph,n,l),n) for n in graph.nodes()]
            heapify(CIs)
            influencers = nlargest(ntoremove,CIs)
            heapify(influencers)

            scores = list(merge(influencers,scores))
            graph.remove_nodes_from([inf[1] for inf in influencers])

            giant = max(nx.connected_component_subgraphs(graph), key=len)
            Gq = giant.number_of_nodes() / N
            removed = N - graph.number_of_nodes()
            q = removed / N

            logging.info('q = {:.3f} | Gq = {:.3f}'.format(q, Gq))

            if removed == N or Gq <= self.CUTOFF :
                break

        return (l,scores)

    def _best_percolation(self, scores_ball):
        scores_ball = [(len(scores),(l,scores)) for l,scores in scores_ball]
        heapify(scores_ball)
        best = nsmallest(1,scores_ball)[0]
        logging.info('Best l:{}, {} nodes out of {} total nodes'.format(best[1][0],best[0],self.G.number_of_nodes()))
        return best[1][1]

    @staticmethod
    def get_ci(graph,n,l):
        return (graph.degree(n)-1) * sum([graph.degree(neighbor)-1 for neighbor,path in nx.single_source_shortest_path(graph, n, l).items() if (len(path)-1) == l])
