from libs.sampling.node_ranking import NodeRanking
import networkx as nx
import random

class Triangles(NodeRanking):

    def __init__(self, G, pseeds):
        super(Triangles, self).__init__(G, pseeds)

    def compute_node_scores(self):
        nodes = list(self.G.nodes())
        random.shuffle(nodes)
        triangles = nx.triangles(self.G)
        self.scores = [(triangles[n],n) for n in nodes]

