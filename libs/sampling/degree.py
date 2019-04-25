from libs.sampling.node_ranking import NodeRanking
import networkx as nx
import random

class Degree(NodeRanking):

    def __init__(self, G, pseeds):
        super(Degree, self).__init__(G, pseeds)

    def compute_node_scores(self):
        nodes = list(self.G.nodes())
        random.shuffle(nodes)
        self.scores = [(self.G.degree(n),n) for n in nodes]

