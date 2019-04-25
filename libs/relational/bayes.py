import warnings

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import numpy as np
import pandas as pd
from libs.utils.loggerinit import *
from libs.relational.relationalhandler import Relational
from scipy.misc import logsumexp

class Bayes(Relational):

    def __init__(self, LC, G, Gseeds, label, RCattributes, LCattributes, ignore):
        super(Bayes, self).__init__(LC, G,  Gseeds, label, RCattributes, LCattributes, ignore)
        self.cpa = None
        self.cpn = None

    def learn(self):
        self._learn_class_prior()
        self._learn_cond_prob_attributes()
        self._learn_relational()

    def _learn_cond_prob_attributes(self):

        # cond. prob. attribute-value|label
        if self.LCattributes:

            labels = self.classprior.index.values
            self.cpa = {attribute: pd.DataFrame(np.zeros((len(labels))), index=labels, columns=[self.NULL]) for attribute in self.G.graph['attributes']}

            for node in self.Gseeds.nodes(data=True):
                label = node[1][self.label]

                for attribute, value in node[1].items():

                    if attribute != self.label:

                        if value == self.ignore:
                            continue

                        col = value

                        if col not in self.cpa[attribute].columns:
                            self.cpa[attribute] = self.cpa[attribute].join(pd.DataFrame(np.zeros((len(labels))), index=labels,columns=[col]))

                        self.cpa[attribute].loc[label,col] += 1.


            for k,df in self.cpa.items():

                # for missing attributes in some classes only in training
                self.cpa[k] += self.smoothing

                # prob.
                self.cpa[k] = self.cpa[k].div( self.cpa[k].sum(axis=1), axis=0 )

    def _learn_relational(self):
        labels = self.classprior.index.values
        columns = ['{}-{}'.format(l,l) for l in labels]
        columns.extend([self.NULL])

        if self.RCattributes:
            self.cpn = {attribute:pd.DataFrame(np.zeros((len(labels),len(columns))), index=labels, columns=columns) for attribute in self.G.graph['attributes']}
        else:
            self.cpn = {self.label: pd.DataFrame(np.zeros((len(labels),len(columns))), index=labels, columns=columns)}

        for node in self.Gseeds.nodes(data=True):
            label = node[1][self.label]

            for neighbor in self.Gseeds[node[0]].keys():
                label_neighbor = self.Gseeds.node[neighbor][self.label]

                for attribute,value in self.Gseeds.node[neighbor].items():

                    if attribute != self.label and not self.RCattributes:
                        continue

                    if value == self.ignore:
                        continue

                    col = '{}-{}'.format(label_neighbor, value)
                    if col not in self.cpn[attribute].columns:
                        self.cpn[attribute] = self.cpn[attribute].join(pd.DataFrame(np.zeros((len(labels))), index=labels, columns=[col]))

                    self.cpn[attribute].loc[label, col] += 1.

        # removing the dummy variable for class label
        self.cpn[self.label].drop(columns=[self.NULL],inplace=True)

        for k,df in self.cpn.items():

            # for missing neighbor-attribute in some classes only in training
            self.cpn[k] += self.smoothing

            # probs.
            # normalized row-wise (WWW18)
            # row_id: node
            # col_id: neighbor
            '''
            # Probability of node vi being xi(color) c  having neighbors Ni is equal
            # to the Probability of the Neighbors Ni given node vi being c times probability of being c 
            P(xi=c|Ni) = P(Ni|c)P(c) / P(Ni) ~ P(Ni|c)P(c)

            # where P(Ni|c)
            # the product of probabilities of neighbors
            # Probability of a neighbor being color <xj> having a node color c = P(Neighbor-of-color-c|color-c)
            P(Ni|c) = 1/Z Product_vj\inNi P(xj=<xj>|xi=c)^wij
            '''
            self.cpn[k] = self.cpn[k].div(self.cpn[k].sum(axis=1), axis=0)

            # # normalized col-wise (it is not like this)
            # self.cpn[k] = self.cpn[k].div(self.cpn[k].sum(axis=0), axis=1)


    def init(self, n):
        if self.LC == 'prior':
            return self.classprior.copy() # init only with prior
        elif self.LC == 'relational':
            return self.classify(n)  # init using neighbors info
        elif self.LC == 'random':
            tmp = (self.classprior.copy() * 0) + 1
            return tmp / tmp.sum() # always random
        elif self.LC == 'minority':
            tmp = (self.classprior.copy() * 0) + 1
            tmp[np.argmin(self.classprior)] += 10
            return tmp / tmp.sum() # always random

    def classify(self, n):
        '''https://stats.stackexchange.com/questions/105602/example-of-how-the-log-sum-exp-trick-works-in-naive-bayes'''

        prob = self.classprior.copy() * 0

        # cond. prob. attribute-value|label
        if self.LCattributes:
            for attribute in self.G.graph['attributes']:
                if attribute != self.label:
                    value = self.G.node[n][attribute]
                    col = value

                    if col not in self.cpa[attribute].columns:
                        col = self.NULL

                    try:
                        prob = prob.add( np.log(self.cpa[attribute][col]) )
                    except:
                        logging.error('This shouldnt happen {}-{} not in cpa'.format(attribute, value))
                        pass

        # if no neighbors then only local info (class prior)
        if len(self.G[n]) == 0:
            return np.exp(prob.subtract(logsumexp(prob)))

        # cond. prob. neighbor-attribute-value|label
        nones = 0
        neighbors = list(self.G[n].keys())
        if self.MAXNEIGHBORS is not None:
            np.random.shuffle(neighbors)
            neighbors = neighbors[:self.MAXNEIGHBORS]

        # computing P(Ni|c)
        for neighbor in neighbors:
            label = 'pred' if neighbor not in self.Gseeds else self.label

            if label == 'pred' and label not in self.G.node[neighbor] and 'prob' not in self.G.node[neighbor]:
                continue

            elif label == 'pred' and label not in self.G.node[neighbor] and 'prob' in self.G.node[neighbor] and self.G.node[neighbor]['prob'] is None:
                # prob can be None when CI iterative
                nones += 1

            elif label == 'pred' and label not in self.G.node[neighbor] and 'prob' in self.G.node[neighbor] and self.G.node[neighbor]['prob'] is not None:
                # when pred doesnt exit but prob is CI relaxation (first round)
                if np.any(self.G.node[neighbor]['prob'] == 0):
                    logging.warning('Probability zero: node {}, neighbor {} | prob: \n{}'.format(n,neighbor,self.G.node[neighbor]['prob']))

                    p = self.G.node[neighbor]['prob']
                    p += 0.0000001
                    p /= p.values.sum()

                    logging.warning('New Probability vector: prob: \n{}'.format(p))
                else:
                    p = self.G.node[neighbor]['prob']

                prob = prob.add( np.log( p ) )

            elif label in self.G.node[neighbor]:
                label = self.G.node[neighbor][label]

                if self.RCattributes:

                    for attribute in self.G.graph['attributes']:

                        if attribute not in self.G.node[neighbor]:
                            logging.warning('can this happen? {} not in {}'.format(attribute,neighbor))
                            pass

                        value = self.G.node[neighbor][attribute]
                        col = '{}-{}'.format(label,value)

                        if col not in self.cpn[attribute].columns:
                            col = self.NULL

                        try:
                            prob = prob.add( np.log( self.cpn[attribute][col] ) )
                        except:
                            logging.error('This shouldnt happen {}-{}-{} not in cpn'.format(label, attribute, value))
                            pass

                else:
                    col = '{}-{}'.format(label, label)
                    if col not in self.cpn[self.label].columns:
                        col = self.NULL

                    try:
                        prob = prob.add(np.log(self.cpn[self.label][col]))
                    except:
                        logging.error('This shouldnt happen {}-{}-{} not in cpn'.format(label, self.label, col))
                        pass


            else:
                logging.error('this is weird. bayes classify n:{}, no iter , no relax, no label in neighbor'.format(n))

        if nones == len(neighbors):
            return None

        # normalizing P(Ni|c)
        prob = prob.subtract(logsumexp(prob))

        # class prior P(c)
        prob = prob.add( np.log(self.classprior.copy()) )

        # eliminating log
        prob = np.exp( prob )

        # normalizing
        prob = prob / prob.sum()
        return prob



# def initialize_label(self, n):
    #     if self.LC == 'attributes':
    #         prob = self.classprior.copy() * 0
    #         prob += 1.
    #
    #         for attribute in self.G.graph['attributes']:
    #             if attribute != self.label:
    #                 value = self.G.node[n][attribute]
    #                 col = value
    #
    #                 if col not in self.cpa[attribute].columns:
    #                     col = self.NULL
    #
    #                 try:
    #                     prob = prob.multiply( self.cpa[attribute][col] )
    #                 except:
    #                     logging.error('This shouldnt happen {}-{} not in cpa'.format(attribute, value))
    #                     pass
    #
    #         if np.all(prob == 1.):
    #             logging.error('This shouldnt happen cpa all 1')
    #             prob = self.classprior.copy()
    #         else:
    #             prob /= prob.sum()
    #     else:
    #         prob = self.classprior.copy()
    #     return prob