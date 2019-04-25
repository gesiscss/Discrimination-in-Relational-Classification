from libs.utils.loggerinit import *
import sys

class NetworkSample(object):

    SNOWBALL = 'SB'
    LIGHTWEIGHTRANDOMWALK = 'LWRW'
    FRIENDSHIPPARADOX = 'FP'
    EGO = 'EGO'
    RANDOMWALK = 'RANDOMWALK'

    def __init__(self, G, pseeds):
        self.G = G
        self.pseeds = pseeds
        self.nseeds = int(round(self.G.number_of_nodes() * self.pseeds))
        self.sample = None
        self.Gsample = None

    def compute_sample(self):
        return

    @staticmethod
    def get_instance(method, G, pseeds):

        from libs.sampling.lightweight_randomwalk import LightWeightRandomWalk
        from libs.sampling.snowball import Snowball
        from libs.sampling.friendship_paradox import FriendshipParadox
        from libs.sampling.ego import Ego
        from libs.sampling.random_walk import RandomWalk

        if method == NetworkSample.LIGHTWEIGHTRANDOMWALK:
            return LightWeightRandomWalk(G, pseeds)

        elif method == NetworkSample.SNOWBALL:
            return Snowball(G, pseeds)

        elif method == NetworkSample.FRIENDSHIPPARADOX:
            return FriendshipParadox(G, pseeds)

        elif method == NetworkSample.EGO:
            return Ego(G, pseeds)

        elif method == NetworkSample.RANDOMWALK:
            return RandomWalk(G, pseeds)