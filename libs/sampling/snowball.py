from libs.sampling.network_sample import NetworkSample
import networkx as nx
import numpy as np
import sys
from libs.utils.loggerinit import *

class Snowball(NetworkSample):

    def __init__(self, G, pseeds):
        super(Snowball, self).__init__(G, pseeds)

    def compute_sample(self):
        if self.pseeds is None:
            logging.error('pseeds is None.')
            sys.exit(0)

        sample = set()
        g_vertices = self.G.nodes()
        np.random.shuffle(g_vertices)
        n_samples = int(round(self.pseeds * len(g_vertices)))
        print('sampling snowball: {} ({}%) nodes'.format(n_samples, self.pseeds))
        i = 0
        steps = 2
        while True:
            i += 1
            x = np.random.choice(g_vertices)
            sample.add(x)
            todo = [x]
            for step in range(steps):
                tmp = []
                for current in todo:
                    neighbors = [n for n in self.G[current]]
                    np.random.shuffle(neighbors)
                    for neighbor in neighbors:
                        sample.add(neighbor)
                        tmp.append(neighbor)
                        if len(sample) >= n_samples:
                            self.sample = list(sample)
                            return
                todo = tmp
        self.sample = list(sample)

    def get_seednodes(self):
        return self.sample