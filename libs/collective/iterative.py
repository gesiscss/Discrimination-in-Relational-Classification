from libs.collective.collectivehandler import Collective
import numpy as np
from collections import Counter
from collections import Counter

import numpy as np

from libs.collective.collectivehandler import Collective
from libs.utils.loggerinit import *
import networkx as nx

class Iterative(Collective):
    T = 1000

    def __init__(self,relational,test):
        super(Iterative, self).__init__(relational,test)

        if self.test:
            self.T = 20

    def learn(self):
        self.relational.learn()

    def classify(self):

        nodes = [n for n in self.relational.G.nodes() if n not in self.relational.Gseeds.nodes()]

        # 1. Init every node with local classifier
        prob = {n: None for n in nodes}
        nx.set_node_attributes(self.relational.G, 'prob', prob)

        # 2. shuffle nodes
        np.random.shuffle(nodes)

        # 3. Estimate label
        for t in range(self.T):
            for n in nodes:
                self.relational.G.node[n]['prob'] = self.relational.classify(n)
                if self.relational.G.node[n]['prob'] is None:
                    continue
                xn = np.argmax(self.relational.G.node[n]['prob'])
                self.relational.G.node[n]['pred'] = xn

        nones = sum([1 for n in nodes if self.relational.G.node[n]['prob'] is None])
        if nones > 0:
            logging.warning('There are {} nodes with None probabilities'.format(nones))