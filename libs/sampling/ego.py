from libs.sampling.network_sample import NetworkSample
import networkx as nx
import numpy as np
import sys
from libs.utils.loggerinit import *

'''
Picks a node randomly and all its 1HOP neighborhood (ego network)
'''
class Ego(NetworkSample):

    def __init__(self, G, pseeds):
        super(Ego, self).__init__(G, pseeds)

    def compute_sample(self, version):

        if self.pseeds is None:
            logging.error('pseeds is None.')
            sys.exit(0)

        g_vertices = list(self.G.nodes())
        np.random.shuffle(g_vertices)
        n_samples = int(round(self.pseeds * len(g_vertices)))
        print('sampling ego: {} ({}%) nodes'.format(n_samples, self.pseeds))

        if version in [1,'ego']:
            return self._compute_sample_version_ego(g_vertices, n_samples, True)
        elif version in [2,'1hop']:
            return self._compute_sample_version_ego(g_vertices, n_samples, False)
        elif version in [3,'1hopeven']:
            return self._compute_sample_version_ego(g_vertices, n_samples, False, True)
        elif version in [4,'1hopodd']:
            return self._compute_sample_version_ego(g_vertices, n_samples, False, False)

    def _compute_sample_version_ego(self, g_vertices, n_samples, center, even=None):
        sample = set()

        while True:
            x = np.random.choice(g_vertices)

            if center:
                sample.add(x)

            neighbors = set([n for n in self.G[x]])

            if even is None:
                sample |= neighbors
            else:
                if len(neighbors) % 2 == 0 and even:
                    sample |= neighbors
                elif len(neighbors) % 2 != 0 and not even:
                    sample |= neighbors

            if len(sample) >= n_samples:
                self.sample = list(sample)
                return

    def get_seednodes(self):
        return self.sample