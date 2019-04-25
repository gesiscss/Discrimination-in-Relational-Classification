from heapq import *

class NodeRanking(object):

    OPTIMALPERCOLATION = 'OP'
    DEGREE = 'D'
    PAGERANK = 'PR'
    TRIANGLES = 'TR'

    def __init__(self, G, pseeds):
        self.G = G
        self.pseeds = pseeds
        self.nseeds = int(round(self.G.number_of_nodes() * self.pseeds))
        self.scores = None

    def compute_node_scores(self):
        return

    def rank_nodes(self, asc=False):
        heapify(self.scores)
        if asc is None:
            asc = [n for s, n in nsmallest(len(self.scores), self.scores)]
            top = int(round(self.nseeds / 2.,0))
            bottom = self.nseeds - top
            tmp = asc[:top]
            tmp.extend(asc[len(asc)-bottom:])
            return tmp
        elif asc:
            return [n for s, n in nsmallest(len(self.scores), self.scores)][:self.nseeds]
        return  [n for s,n in nlargest(len(self.scores),self.scores)][:self.nseeds]

    @staticmethod
    def get_instance(method, G, pseeds):

        from libs.sampling.optimal_percolation import OptimalPercolation
        from libs.sampling.degree import Degree
        from libs.sampling.page_rank import PageRank
        from libs.sampling.triangles import Triangles

        if method == NodeRanking.OPTIMALPERCOLATION:
            return OptimalPercolation(G, pseeds)

        elif method == NodeRanking.DEGREE:
            return Degree(G,pseeds)

        elif method == NodeRanking.PAGERANK:
            return PageRank(G,pseeds)

        elif method == NodeRanking.TRIANGLES:
            return Triangles(G,pseeds)