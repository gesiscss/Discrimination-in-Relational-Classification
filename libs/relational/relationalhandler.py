
import sys

import numpy as np
import pandas as pd

from libs.utils.loggerinit import *


class Relational(object):

    NULL = 'null'
    MAXNEIGHBORS = None

    def __init__(self, LC, G, Gseeds, label, RCattributes, LCattributes, ignore=None):
        self.LC = LC
        self.G = G
        self.Gseeds = Gseeds
        self.label = label
        self.RCattributes = RCattributes
        self.LCattributes = LCattributes
        self.ignore = ignore
        self.classprior = None
        self.attributes = list(self.Gseeds.graph['attributes'])
        self.attributes.remove(self.label)
        self.smoothing = 1.

    @staticmethod
    def get_instance(LC, RC, G, Gseeds, label, RCattributes, LCattributes, ignore=None):
        from libs.relational.bayes import Bayes

        if RC == 'bayes':
            return Bayes(LC, G, Gseeds, label, RCattributes, LCattributes, ignore)
        else:
            logging.error('RC:{} not implemented yet.'.format(RC))
            sys.exit(0)

    def learn(self):
        return

    def _learn_class_prior(self):
        size = len(self.Gseeds.graph['labels'])
        self.classprior = pd.Series(np.zeros((size)),index=self.Gseeds.graph['labels'])

        for label in self.Gseeds.graph['labels']:
            self.classprior.loc[label] = len([0 for n in self.Gseeds.nodes(data=True) if n[1][self.label]==label])

        self.classprior += self.smoothing
        self.classprior /= self.classprior.sum()

    def init(self, n):
        return

    def classify(self, n):
        return

    def initialize_label(self,n):
        return