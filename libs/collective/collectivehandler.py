import sys
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score

from libs.utils.loggerinit import *
from sklearn.metrics import mean_squared_error
from collections import Counter

class Collective(object):

    NDRAWS = 1

    def __init__(self, relational,test):
        self.relational = relational
        self.test = test

    @staticmethod
    def get_instance(CI,relational,test):
        from libs.collective.gibbs import Gibbs
        from libs.collective.relaxation import Relaxation
        from libs.collective.iterative import Iterative

        if CI == 'gibbs':
            return Gibbs(relational,test)
        elif CI == 'relaxation':
            return Relaxation(relational,test)
        elif CI == 'iterative':
            return Iterative(relational,test)
        else:
            logging.error('CI:{} not implemented yet.'.format(CI))
            sys.exit(0)

    def learn(self):
        return

    def classify(self):
        return

    def evaluation(self):

        nodes = [n for n in self.relational.G if n not in self.relational.Gseeds and 'pred' in self.relational.G.node[n] and 'prob' in self.relational.G.node[n] and self.relational.G.node[n]['pred'] is not None and self.relational.G.node[n]['prob'] is not None]
        unknowns = self.relational.G.number_of_nodes() - self.relational.Gseeds.number_of_nodes()
        missing = unknowns - len(nodes)

        nones = [n for n in self.relational.G if n not in self.relational.Gseeds and n not in nodes and self.relational.G.node[n]['prob'] is None]
        nans = missing - len(nones)

        if missing > 0:
            logging.warning('{} (out of {}) nodes were not classified. They still remain unknown.'.format(missing, unknowns))
        if nans > 0:
            logging.warning('- From the missing ones: {} (out of {}) due to overflow.'.format(nans, missing))
        if len(nones) > 0:
            logging.warning('- From the missing ones: {} (out of {}) due to prob NONE.'.format(len(nones), missing))

        y_true = [self.relational.G.node[n][self.relational.label] for n in nodes]
        y_pred = [self.relational.G.node[n]['pred'] for n in nodes]
        y_prob = [self.relational.G.node[n]['prob'] for n in nodes]
        labels = self.relational.Gseeds.graph['labels']

        for label in labels:
            if label not in y_true:
                logging.warning('{} not in y_true'.format(label))

        accuracy = Collective._get_accuracy(y_true,y_pred,labels)
        f1 = Collective._get_f1(y_true, y_pred,labels)
        mae = Collective._get_mae(y_true, y_pred, labels)
        rmse = Collective._get_rmse(y_true, y_pred, labels)
        cnf_matrix = Collective._get_confusion_matrix(y_true, y_pred, labels)
        precision_recall, average_precision = Collective._get_precision_recall(y_true, y_prob, labels)
        roc_auc = Collective._get_roc_auc(y_true, y_prob, labels)
        error = Collective._get_error(y_true, y_pred, labels)

        return labels, nodes, accuracy, f1, mae, rmse, cnf_matrix, [precision_recall,average_precision], roc_auc, error, [labels.index(y) for y in y_true], [labels.index(y) for y in y_pred], y_prob

    @staticmethod
    def _get_error(y_true, y_pred, labels):
        error = {}
        try:
            _y_true = np.array([labels.index(y) for y in y_true])
            _y_pred = np.array([labels.index(y) for y in y_pred])
            error['overall'] = sum(_y_true!=_y_pred)/float(_y_true.size)
        except Exception as ex:
            logging.error(ex)
            print(ex)
            pass

        tmp_true = Counter(_y_true)
        logging.info('true counts: {}'.format(tmp_true))

        try:
            for label in labels:
                if label not in y_true:
                    continue
                _y_true = np.array([int(y==label) for y in y_true])
                _y_pred = np.array([int(y==label) for y in y_pred])
                m = np.multiply(_y_true,_y_pred).sum()
                logging.info('correct {}: {}'.format(label,m.sum()))
                error[label] = (_y_true.sum()-m)/_y_true.sum()
        except:
            pass
        return error

    @staticmethod
    def _get_accuracy(y_true,y_pred,labels):
        accuracy = {}
        try:
            _y_true = [labels.index(y) for y in y_true]
            _y_pred = [labels.index(y) for y in y_pred]
            accuracy['overall'] = accuracy_score(_y_true, _y_pred)
        except:
            pass
        return accuracy


    @staticmethod
    def _get_f1(y_true,y_pred, labels):
        # 'weighted': Calculate metrics for each label, and find their average,
        # weighted by support (the number of true instances for each label).
        # This alters ‘macro’ to account for label imbalance; it can result
        # in an F-score that is not between precision and recall.
        f1 = {}
        try:
            _y_true = [labels.index(y) for y in y_true]
            _y_pred = [labels.index(y) for y in y_pred]
            f1['overall'] = f1_score(_y_true, _y_pred, average='weighted')
        except:
            pass
        return f1

    @staticmethod
    def _get_mae(y_true, y_pred, labels):
        mae = {}
        try:
            _y_true = [labels.index(y) for y in y_true]
            _y_pred = [labels.index(y) for y in y_pred]
            mae['overall'] = mean_absolute_error(_y_true, _y_pred)
        except:
            pass
        return mae

    @staticmethod
    def _get_rmse(y_true, y_pred, labels):
        rmse = {}
        try:
            _y_true = [labels.index(y) for y in y_true]
            _y_pred = [labels.index(y) for y in y_pred]
            rmse['overall'] = np.sqrt(mean_squared_error(_y_true, _y_pred))
        except:
            pass
        return rmse

    @staticmethod
    def _get_confusion_matrix(y_true, y_pred, labels):
        try:
            return confusion_matrix(y_true, y_pred, labels)
        except:
            return None

    @staticmethod
    def _get_precision_recall(y_true, y_prob, labels):
        precision_recall = {}
        average_precision = {}
        for label in labels:
            if label not in y_true:
                continue
            _y_true = [int(y==label) for y in y_true]
            _y_prob = [y.loc[label] for y in y_prob]
            try:
                precision_recall[label] = precision_recall_curve(_y_true,_y_prob)
                average_precision[label] = average_precision_score(_y_true,_y_prob)
            except:
                pass
        return precision_recall,average_precision

    @staticmethod
    def _get_roc_auc(y_true, y_prob, labels):
        roc_auc = {}
        for label in labels:
            if label not in y_true:
                continue
            _y_true = [int(y == label) for y in y_true]
            _y_prob = [y.loc[label] for y in y_prob]
            try:
                roc_auc[label] = roc_auc_score(_y_true, _y_prob, 'weighted')
            except:
                pass
        return roc_auc