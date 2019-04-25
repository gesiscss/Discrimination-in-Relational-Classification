from libs.collective.collectivehandler import Collective
import numpy as np
from collections import Counter
from collections import Counter

import numpy as np

from libs.collective.collectivehandler import Collective
import networkx as nx

class Gibbs(Collective):
    BURNIN = 100
    NUMIT = 1000

    def __init__(self,relational,test):
        super(Gibbs, self).__init__(relational,test)

        if self.test:
            self.BURNIN = 5
            self.NUMIT = 20

    def learn(self):
        self.relational.learn()

    def classify(self):

        nodes = [n for n in self.relational.G.nodes() if n not in self.relational.Gseeds.nodes()]

        # 1. Init every node with local classifier (only known neighbors)
        prob = {n:self.relational.classify(n) for n in nodes} ## <--------- check this, maybe local means different
        nx.set_node_attributes(self.relational.G, 'prob', prob)

        pred = {n: self.relational.Gseeds.graph['labels'][np.argmax(np.random.multinomial(self.NDRAWS, self.relational.G.node[n]['prob'], size=1))] for n in nodes}
        nx.set_node_attributes(self.relational.G, 'pred', pred)

        empty = {n:[] for n in nodes}
        nx.set_node_attributes(self.relational.G, 'assignments', empty)

        # 2. shuffle nodes
        np.random.shuffle(nodes)

        # 3. Burning phase
        for step in range(self.BURNIN):
            for n in nodes:
                self.relational.G.node[n]['prob'] = self.relational.classify(n)
                tosses = np.random.multinomial(self.NDRAWS, self.relational.G.node[n]['prob'], size=1)
                self.relational.G.node[n]['pred'] = self.relational.Gseeds.graph['labels'][np.argmax(tosses)]

        # 4. Burning phase
        for step in range(self.NUMIT):
            for n in nodes:
                self.relational.G.node[n]['prob'] = self.relational.classify(n)
                tosses = np.random.multinomial(self.NDRAWS, self.relational.G.node[n]['prob'], size=1)
                self.relational.G.node[n]['assignments'].append(self.relational.Gseeds.graph['labels'][np.argmax(tosses)])

        # 5. Norm counts. -> final class prob. estimates
        for n in nodes:
            counts = Counter(self.relational.G.node[n]['assignments'])

            self.relational.G.node[n]['prob'] = self.relational.classprior.copy() * 0
            for label in self.relational.Gseeds.graph['labels']:
                self.relational.G.node[n]['prob'].loc[label] = counts[label] + self.relational.smoothing

            self.relational.G.node[n]['prob'] /= self.relational.G.node[n]['prob'].sum()
            tosses = np.random.multinomial(self.NDRAWS, self.relational.G.node[n]['prob'], size=1)
            self.relational.G.node[n]['pred'] = self.relational.Gseeds.graph['labels'][np.argmax(tosses)]
