from libs.collective.collectivehandler import Collective
import numpy as np
from collections import Counter
from collections import Counter

import numpy as np
import networkx as nx
from libs.collective.collectivehandler import Collective


class Relaxation(Collective):
    T = 99
    k = 1.
    alpha = 0.99

    def __init__(self,relational,test):
        super(Relaxation, self).__init__(relational,test)

        if self.test:
            self.T = 20

    def learn(self):
        self.relational.learn()

    def classify(self):

        nodes = [n for n in self.relational.G.nodes() if n not in self.relational.Gseeds.nodes()]

        # 1. Init every node
        prob = {n: self.relational.init(n) for n in nodes}  # init only with prior
        nx.set_node_attributes(G=self.relational.G, name='prob', values=prob)

        # 2. Estimate label
        betas = [self.k]
        for t in range(self.T):

            betas.append(betas[-1] * self.alpha)

            c_n = {n: betas[t + 1] * self.relational.classify(n) + (1 - betas[t + 1]) * self.relational.G.node[n]['prob'] for n in nodes}
            nx.set_node_attributes(G=self.relational.G, name='prob', values=c_n)

            x_n = {n:self.relational.Gseeds.graph['labels'][np.argmax(np.random.multinomial(self.NDRAWS, self.relational.G.node[n]['prob'], size=1))] for n in nodes}
            nx.set_node_attributes(G=self.relational.G, name='pred', values=x_n)
