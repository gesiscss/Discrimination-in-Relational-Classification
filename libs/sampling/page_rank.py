from libs.sampling.node_ranking import NodeRanking
import networkx as nx
import random

class PageRank(NodeRanking):

    def __init__(self, G, pseeds):
        super(PageRank, self).__init__(G, pseeds)

    def compute_node_scores(self):
        nodes = list(nx.pagerank(self.G).items())
        random.shuffle(nodes)
        self.scores = [(pr,n) for n,pr in nodes]
