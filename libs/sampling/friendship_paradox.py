from libs.sampling.network_sample import NetworkSample
import networkx as nx
import numpy as np
import sys
from libs.utils.loggerinit import *

'''
Beta version.
Still in testing mode
'''
class FriendshipParadox(NetworkSample):

    def __init__(self, G, pseeds):
        super(FriendshipParadox, self).__init__(G, pseeds)

    def compute_sample(self, version):

        if version not in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]:
            logging.error('version {} not implemented yet.'.format(version))
            sys.exit(0)

        if self.pseeds is None:
            logging.error('pseeds is None.')
            sys.exit(0)

        g_vertices = list(self.G.nodes())
        np.random.shuffle(g_vertices)
        n_samples = int(round(self.pseeds * len(g_vertices)))
        print('sampling friendship paradox: {} ({}%) nodes'.format(n_samples, self.pseeds))

        if version == 1:
            return self._compute_sample_version_1(g_vertices, n_samples)
        if version == 2:
            return self._compute_sample_version_2(g_vertices, n_samples)
        if version == 3:
            return self._compute_sample_version_3(g_vertices, n_samples)
        if version == 4:
            return self._compute_sample_version_4(g_vertices, n_samples)
        if version == 5:
            return self._compute_sample_version_5(g_vertices, n_samples)
        if version == 6:
            return self._compute_sample_version_6(g_vertices, n_samples)
        if version == 7:
            return self._compute_sample_version_7(g_vertices, n_samples)
        if version == 8:
            return self._compute_sample_version_8(g_vertices, n_samples)
        if version == 9:
            return self._compute_sample_version_9(g_vertices, n_samples)
        if version == 10:
            return self._compute_sample_version_10(g_vertices, n_samples)
        if version == 11:
            return self._compute_sample_version_11(g_vertices, n_samples)
        if version == 12:
            return self._compute_sample_version_12(g_vertices, n_samples)
        if version == 13:
            return self._compute_sample_version_13(g_vertices, n_samples)
        if version == 14:
            return self._compute_sample_version_14(g_vertices, n_samples)
        if version == 15:
            return self._compute_sample_version_15(g_vertices, n_samples)


    def _compute_sample_version_1(self, g_vertices, n_samples):
        '''
        i --> j
        :param g_vertices:
        :param n_samples:
        :return:
        '''
        sample = set()
        while True:
            i = np.random.choice(g_vertices)

            neighbors = [n for n in nx.neighbors(self.G,i)]
            if len(neighbors) == 0:
                continue
            np.random.shuffle(neighbors)
            j = neighbors[0]

            sample.add(i)
            sample.add(j)

            if len(sample) >= n_samples:
                self.sample = list(sample)
                return

    def _compute_sample_version_2(self, g_vertices, n_samples):
        '''
        i --> j --> k
        :param g_vertices:
        :param n_samples:
        :return:
        '''
        sample = set()
        while True:
            i = np.random.choice(g_vertices)

            neighbors = [n for n in nx.neighbors(self.G,i)]
            if len(neighbors) == 0:
                continue
            np.random.shuffle(neighbors)
            j = neighbors[0]

            neighbors = [n for n in nx.neighbors(self.G,j) if n!=i]
            if len(neighbors) == 0:
                continue
            np.random.shuffle(neighbors)
            k = neighbors[0]

            sample.add(i)
            sample.add(j)
            sample.add(k)

            if len(sample) >= n_samples:
                self.sample = list(sample)
                return

    def _compute_sample_version_3(self, g_vertices, n_samples):
        '''
        i --> (j) --> k (j is not included), therefore: i --> k )
        :param g_vertices:
        :param n_samples:
        :return:
        '''
        sample = set()
        while True:
            i = np.random.choice(g_vertices)

            neighbors = [n for n in nx.neighbors(self.G, i)]
            if len(neighbors) == 0:
                continue
            np.random.shuffle(neighbors)
            j = neighbors[0]

            neighbors = [n for n in nx.neighbors(self.G,j) if n != i]
            if len(neighbors) == 0:
                continue
            np.random.shuffle(neighbors)
            k = neighbors[0]

            sample.add(i)
            sample.add(k)

            if len(sample) >= n_samples:
                self.sample = list(sample)
                return

    def _compute_sample_version_4(self, g_vertices, n_samples):
        '''
        (i) --> j (only j)
        :param g_vertices:
        :param n_samples:
        :return:
        '''
        sample = set()
        while True:
            i = np.random.choice(g_vertices)

            neighbors = [n for n in nx.neighbors(self.G,i)]
            if len(neighbors) == 0:
                continue
            np.random.shuffle(neighbors)
            j = neighbors[0]
            sample.add(j)

            if len(sample) >= n_samples:
                self.sample = list(sample)
                return

    def _compute_sample_version_5(self, g_vertices, n_samples):
        '''
        (i) --> (j) --> k (only k)
        :param g_vertices:
        :param n_samples:
        :return:
        '''
        sample = set()
        while True:
            i = np.random.choice(g_vertices)

            neighbors = [n for n in nx.neighbors(self.G,i)]
            if len(neighbors) == 0:
                continue
            np.random.shuffle(neighbors)
            j = neighbors[0]

            neighbors = [n for n in nx.neighbors(self.G,j) if n!=i]
            if len(neighbors) == 0:
                continue
            np.random.shuffle(neighbors)
            k = neighbors[0]

            sample.add(k)

            if len(sample) >= n_samples:
                self.sample = list(sample)
                return

    def _compute_sample_version_6(self, g_vertices, n_samples):
        '''
        j1 <-- (i) --> j2 (j1 and j2) //wedges
        :param g_vertices:
        :param n_samples:
        :return:
        '''
        sample = set()
        while True:
            i = np.random.choice(g_vertices)
            neighbors = [n for n in nx.neighbors(self.G,i)]

            if len(neighbors) > 1:
                np.random.shuffle(neighbors)
                j1 = neighbors[0]
                j2 = neighbors[1]
                sample.add(j1)
                sample.add(j2)

                if len(sample) >= n_samples:
                    self.sample = list(sample)
                    return

    def _compute_sample_version_7(self, g_vertices, n_samples):
        '''
        (i) --> j --> k (i is not included), therefore: j --> k )
        :param g_vertices:
        :param n_samples:
        :return:
        '''
        sample = set()
        while True:
            i = np.random.choice(g_vertices)

            neighbors = [n for n in nx.neighbors(self.G,i)]
            if len(neighbors) == 0:
                continue
            np.random.shuffle(neighbors)
            j = neighbors[0]

            neighbors = [n for n in nx.neighbors(self.G,j) if n!=i]
            if len(neighbors) == 0:
                continue
            np.random.shuffle(neighbors)
            k = neighbors[0]

            sample.add(j)
            sample.add(k)

            if len(sample) >= n_samples:
                self.sample = list(sample)
                return

    def _compute_sample_version_8(self, g_vertices, n_samples):
        '''
        (i) --> (j) ; k1 <-- (j) --> k2 (k1 and k2)
        :param g_vertices:
        :param n_samples:
        :return:
        '''
        sample = set()
        while True:
            i = np.random.choice(g_vertices)
            neighbors = [n for n in nx.neighbors(self.G,i)]

            if len(neighbors) > 0:
                np.random.shuffle(neighbors)
                j = neighbors[0]

                neighbors = [n for n in nx.neighbors(self.G, j) if n!=i]

                if len(neighbors) > 1:
                    np.random.shuffle(neighbors)
                    k1 = neighbors[0]
                    k2 = neighbors[1]

                    if j==i or k1==k2 or k1==i or k2==i or k1==j or k2==j:
                        continue

                    sample.add(k1)
                    sample.add(k2)

                    if len(sample) >= n_samples:
                        self.sample = list(sample)
                        return

    def _compute_sample_version_9(self, g_vertices, n_samples):
        '''
        (i) --> (j) ; k1 <-- j --> k2 (j and k1 and k2)
        :param g_vertices:
        :param n_samples:
        :return:
        '''
        sample = set()
        while True:
            i = np.random.choice(g_vertices)
            neighbors = [n for n in nx.neighbors(self.G, i)]

            if len(neighbors) > 0:
                np.random.shuffle(neighbors)
                j = neighbors[0]

                neighbors = [n for n in nx.neighbors(self.G, j) if n!=i]

                if len(neighbors) > 1:
                    np.random.shuffle(neighbors)
                    k1 = neighbors[0]
                    k2 = neighbors[1]

                    if j == i or k1 == k2 or k1 == i or k2 == i or k1 == j or k2 == j:
                        continue

                    sample.add(j)
                    sample.add(k1)
                    sample.add(k2)

                    if len(sample) >= n_samples:
                        self.sample = list(sample)
                        return

    def _compute_sample_version_10(self, g_vertices, n_samples):
        '''
        j1 <-- i --> j2 (i, j1 and j2) //wedges including random node i (nodes)
        :param g_vertices:
        :param n_samples:
        :return:
        '''
        sample = set()
        while True:
            i = np.random.choice(g_vertices)
            neighbors = [n for n in nx.neighbors(self.G,i)]

            if len(neighbors) > 1:
                np.random.shuffle(neighbors)
                j1 = neighbors[0]
                j2 = neighbors[1]
                sample.add(i)
                sample.add(j1)
                sample.add(j2)

                if len(sample) >= n_samples:
                    self.sample = list(sample)
                    return

    def _compute_sample_version_11(self, g_vertices, n_samples):
        '''
        j1 <-- i --> j2 (i, j1 and j2) //wedges including random node i (edges)
        :param g_vertices:
        :param n_samples:
        :return:
        '''
        sample = set()
        sample_edges = set()
        while True:
            i = np.random.choice(g_vertices)
            neighbors = [n for n in nx.neighbors(self.G,i)]

            if len(neighbors) > 1:
                np.random.shuffle(neighbors)
                j1 = neighbors[0]
                j2 = neighbors[1]

                sample_edges.add(tuple(sorted([i, j1])))
                sample_edges.add(tuple(sorted([i, j2])))
                sample.add(i)
                sample.add(j1)
                sample.add(j2)

                if len(sample) >= n_samples:
                    self.sample = list(sample)
                    self.Gsample = self.G.edge_subgraph(sample_edges)
                    return

    def _compute_sample_version_12(self, g_vertices, n_samples):
        '''
        j1 <-- i --> j2 (i, j1 and j2) //wedges including random node i (edges), even # of nieghbors
        :param g_vertices:
        :param n_samples:
        :return:
        '''
        sample = set()
        seeds = set()
        sample_edges = set()
        while True:
            i = np.random.choice(g_vertices)
            if i in seeds:
                continue
            neighbors = [n for n in nx.neighbors(self.G,i)]
            maxn = np.arange(2,len(neighbors),2)
            maxn = np.append(maxn, [1])
            pick_n_neighbors = np.random.choice(maxn)
            np.random.shuffle(neighbors)

            while pick_n_neighbors > 0:
                j = neighbors.pop(0)
                sample_edges.add(tuple(sorted([i, j])))
                sample.add(i)
                sample.add(j)
                seeds.add(i)
                pick_n_neighbors -= 1

            if len(sample) >= n_samples:
                self.sample = list(sample)
                self.Gsample = self.G.edge_subgraph(sample_edges)
                return


    def _compute_sample_version_13(self, g_vertices, n_samples):
        '''
        wedges3a
        i --> j1
        i --> j2
        i --> j3
        seeds: j1, j2, j3
        :param g_vertices:
        :param n_samples:
        :return:
        '''
        sample = set()
        while True:
            i = np.random.choice(g_vertices)
            neighbors = [n for n in nx.neighbors(self.G,i)]

            if len(neighbors) >= 3:
                np.random.shuffle(neighbors)
                j1 = neighbors[0]
                j2 = neighbors[1]
                j3 = neighbors[2]
                sample.add(j1)
                sample.add(j2)
                sample.add(j3)

                if len(sample) >= n_samples:
                    self.sample = list(sample)
                    return

    def _compute_sample_version_14(self, g_vertices, n_samples):
        '''
        wedges3b
        i --> j1
        i --> j2
        i --> j3
        seeds: i, j1, j2, j3
        :param g_vertices:
        :param n_samples:
        :return:
        '''
        sample = set()
        while True:
            i = np.random.choice(g_vertices)
            neighbors = [n for n in nx.neighbors(self.G, i)]

            if len(neighbors) > 3:
                np.random.shuffle(neighbors)
                j1 = neighbors[0]
                j2 = neighbors[1]
                j3 = neighbors[2]
                sample.add(i)
                sample.add(j1)
                sample.add(j2)
                sample.add(j3)

                if len(sample) >= n_samples:
                    self.sample = list(sample)
                    return

    def _compute_sample_version_15(self, g_vertices, n_samples):
        '''
        friendshipParadoxNSeeds
        i --> j1
        i --> j2
        i --> j2 --> k1
        i --> j2 --> k2
        seeds: j1, k1, k2
        :param g_vertices:
        :param n_samples:
        :return:
        '''
        sample = set()
        while True:
            i = np.random.choice(g_vertices)
            neighbors1hop = [n for n in nx.neighbors(self.G, i)]

            if len(neighbors1hop) >= 2:
                np.random.shuffle(neighbors1hop)
                j1 = neighbors1hop[0]
                j2 = neighbors1hop[1]

                neighbors2hop = [n for n in nx.neighbors(self.G, j2)]
                if len(neighbors2hop) >= 2:
                    np.random.shuffle(neighbors2hop)
                    k1 = neighbors2hop[0]
                    k2 = neighbors2hop[1]

                    sample.add(j1)
                    sample.add(k1)
                    sample.add(k2)

                    if len(sample) >= n_samples:
                        self.sample = list(sample)
                        return

    def get_seednodes(self):
        return self.sample



    # def _compute_sample_version_3(self, g_vertices, n_samples):
    #     '''
    #     i --> (j) --> k (j is not included), therefore: i --> k )
    #     :param g_vertices:
    #     :param n_samples:
    #     :return:
    #     '''
    #     self.Gsample2hop = nx.Graph()
    #     self.Gsample2hop.graph['attributes'] = self.G.graph['attributes']
    #     self.Gsample2hop.graph['name'] = self.G.graph['name']
    #
    #     while True:
    #         i = np.random.choice(g_vertices)
    #
    #         neighbors = [n for n in nx.neighbors(self.G,i)]
    #         if len(neighbors) == 0:
    #             continue
    #         np.random.shuffle(neighbors)
    #         j = neighbors[0]
    #
    #         neighbors = [n for n in nx.neighbors(self.G,j) if n!=i]
    #         if len(neighbors) == 0:
    #             continue
    #         np.random.shuffle(neighbors)
    #         k = neighbors[0]
    #
    #         if not self.Gsample2hop.has_edge(i,k):
    #             self.Gsample2hop.add_edge(i,k)
    #
    #         if self.Gsample2hop.number_of_nodes() >= n_samples:
    #
    #             for a in self.G.graph['attributes']:
    #                 final = {n: v for n, v in dict(nx.get_node_attributes(G=self.G, name=a)).items() if n in list(self.Gsample2hop.nodes())}
    #                 nx.set_node_attributes(G=self.Gsample2hop, name=a, values=final)
    #
    #             self.sample = list(self.Gsample2hop.nodes())
    #             return
