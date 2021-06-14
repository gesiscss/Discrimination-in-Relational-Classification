from __future__ import division

############################################
# Local dependencies
############################################

############################################
# System dependencies
############################################
import pandas as pd
from sklearn.preprocessing import label_binarize
from sklearn import linear_model
import numpy as np
import glob
import networkx as nx
from sklearn import metrics
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedShuffleSplit

############################################
# Class
############################################
class LINK(object):

    def __init__(self):
        '''
        Initializes the network-only Bayes object.
        '''
        self.Gseeds = None
        self.clf = None
        self.feature_x = None
        self.membership_y = None
        self.train_index = None
        self.test_index = None

    def learn(self, Gseeds, feature_x, membership_y, train_index, test_index, test_nodes):
        '''
        Fits the data
        '''
        self.Gseeds = Gseeds
        self.feature_x = feature_x
        self.membership_y = membership_y
        self.train_index = train_index
        self.test_index = test_index
        self.test_nodes = test_nodes

        self.clf = linear_model.LogisticRegression(penalty='l2', C=10e20, solver='lbfgs')
        self.clf.fit(self.feature_x[self.train_index], np.ravel(self.membership_y[self.train_index]))

    def predict(self):
        '''
        LINK: https://github.com/kaltenburger/homophily_monophily_NHB/blob/master/code/functions/LINK.py
        '''
        return self.clf.predict_proba(self.feature_x[self.test_index])

    def info(self):
        '''
        Prints
        '''
        print(self.clf)
