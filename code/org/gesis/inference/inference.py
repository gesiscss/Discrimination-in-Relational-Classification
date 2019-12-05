############################################
# System dependencies
############################################
import os
import time

import pandas as pd
from joblib import Parallel, delayed
############################################
# Local dependencies
############################################
from org.gesis.inference.relaxation import Relaxation
from sklearn.metrics import auc
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from utils.io import create_folder
from utils.io import load_pickle
from utils.io import write_gpickle
from utils.io import write_pickle

############################################
# Constants
############################################
RELAXATION = "relaxation"


############################################
# Functions
############################################

def is_inference_done(root, datafn, sampling, pseeds, postfix):
    output = os.path.join(root, "{}_{}".format(os.path.basename(datafn).replace(".gpickle", ""), sampling))
    f1 = get_graph_filename(output, pseeds, postfix)
    f2 = get_samplegraph_filename(output, pseeds, postfix)
    f3 = get_evaluation_filename(output, pseeds, postfix)

    return os.path.exists(f1) and os.path.exists(f2) and os.path.exists(f3)


def get_graph_filename(output, pseeds, postfix):
    if pseeds < 1:
        pseeds = int(round(pseeds * 100, 1))
    return os.path.join(output, "P{}_graph{}.gpickle".format(pseeds, '_{}'.format(postfix) if postfix is not None else ""))


def get_samplegraph_filename(output, pseeds, postfix):
    if pseeds < 1:
        pseeds = int(round(pseeds * 100, 1))
    return os.path.join(output, "P{}_samplegraph{}.gpickle".format(pseeds, '_{}'.format(postfix) if postfix is not None else ""))


def get_evaluation_filename(output, pseeds, postfix):
    if pseeds < 1:
        pseeds = int(round(pseeds * 100, 1))
    return os.path.join(output, "P{}_evaluation{}.pickle".format(pseeds, '_{}'.format(postfix) if postfix is not None else ""))


def _load_pickle_to_dataframe(fn, verbose=True):
    obj = load_pickle(fn, verbose)
    columns = ['kind', 'N', 'm', 'B', 'H', 'i', 'x', 'sampling', 'pseeds', 'epoch', 'n', 'e', 'min_degree', 'rocauc', 'mae', 'ccm', 'ccM', 'bias', 'lag']

    df = pd.DataFrame({'kind': fn.split("/")[-2].split("-")[0].split("_")[0],
                       'N': int(obj['N']),
                       'm': int(obj['m']),
                       'B': float(obj['B']),
                       'H': float(obj['H']),
                       'i': obj['i'] if obj['i'] is None else int(obj['i']),
                       'x': obj['x'] if obj['x'] is None else int(obj['x']),
                       'sampling': obj['sampling'],
                       'pseeds': float(obj['pseeds']),
                       'epoch': int(obj['epoch']),
                       'n': obj['n'],
                       'e': obj['e'],
                       'min_degree': obj['min_degree'],
                       'rocauc': obj['rocauc'],
                       'mae': obj['mae'],
                       'ccm': obj['ccm'],
                       'ccM': obj['ccM'],
                       'bias': obj['bias'],
                       'lag': obj['lag']}, columns=columns, index=[0])
    return df


def _update_pickle_to_dataframe(fn, verbose=True):
    # /results-individual/BAH-N2000-m4-B0.5-H0.8-i1-x1-h0.8-k8.0-km8.0-kM7.9_nedges/P80_evaluation.pickle

    obj = load_pickle(fn, verbose)

    if fn.split("/")[-2].startswith("BAH"):
        if 'E' in obj:
            N = int(fn.split("/")[-2].split("-")[1][1:])
            m = int(fn.split("/")[-2].split("-")[2][1:])

            obj['n'] = int(obj['N'])
            obj['e'] = int(obj['E'])
            obj['min_degree'] = int(obj['m'])

            obj['N'] = N
            obj['m'] = m
            del (obj['E'])

            write_pickle(obj, fn)
        else:
            if verbose:
                print('{} passed.'.format(fn))
                print(obj)
    else:
        obj['n'] = obj['N']
        obj['e'] = obj['E']
        obj['min_degree'] = obj['m']
        del (obj['E'])
        write_pickle(obj, fn)


############################################
# Class
############################################
class Inference(object):
    def __init__(self, method):
        '''
        Initializes the inference object
        - method: collective inference method
        '''
        self.G = None
        self.Gseeds = None
        self.method = method
        self.rocauc = None
        self.mae = None
        self.ccm = None
        self.ccM = None
        self.fpr = None
        self.tpr = None
        self.duration = None

    def predict(self, G, local_model, relational_model):
        '''
        Creates a new instance of the respective collective inference algorithm
        - G: global network
        - local_model: instance of LC
        - relational_model: instance of RC
        '''
        self.G = G
        self.Gseeds = relational_model.Gseeds

        start_time = time.time()
        if self.method == RELAXATION:
            Relaxation(G,
                       local_model,
                       relational_model).predict()
        else:
            raise Exception("inference method does not exist: {}".format(self.method))
        self.duration = time.time() - start_time

    def evaluation(self):
        '''
        Computes global and group evaluation metrics.
        global: roauc, mae
        group: ccm, ccM, bias
        '''
        labels = self.G.graph['labels']
        y_true, y_score, y_pred = zip(*[(labels.index(self.G.node[n][self.G.graph['class']]),
                                         self.G.node[n]['ci'].loc[labels[1]],
                                         labels.index(self.G.node[n]['xi']))
                                        for n in self.G.nodes() if not self.G.node[n]['seed']])

        print(y_true[:5])
        print(y_pred[:5])
        print(y_score[:5])

        # general performance metrics
        self.rocauc = roc_auc_score(y_true, y_score)
        self.fpr, self.tpr, _ = roc_curve(y_true, y_score)
        self.rocauc_curve = auc(self.fpr, self.tpr)
        self.mae = mean_absolute_error(list(y_true), list(y_pred))

        # minority performance
        m = sum(y_true)
        self.ccm = sum([1 for t, p in zip(*[y_true, y_pred]) if t == p and t == 1]) / m

        # majority performance
        M = len(y_true) - m
        self.ccM = sum([1 for t, p in zip(*[y_true, y_pred]) if t == p and t == 0]) / M

        # fairness
        self.bias = self.ccm / (self.ccm + self.ccM)

    def summary(self):
        '''
        Prints evaluation metric values
        '''
        print("")
        print("Prediction in {} seconds".format(self.duration))
        print("ROCAUC: {}".format(self.rocauc))
        print("ROCAUC curve: {}".format(self.rocauc_curve))
        print("MAE: {}".format(self.mae))
        print("ccm: {}".format(self.ccm))
        print("ccM: {}".format(self.ccM))
        print("bias: {}".format(self.bias))
        print("")

    def save(self, root, postfix=None):
        tmp = os.getcwd()

        # create folder for network and pseeds
        create_folder(root, True)
        try:
            output = self.G.graph['fullname']
        except:
            output = self.G.graph['name']

        output = "{}_{}".format(output, self.Gseeds.graph["method"])
        create_folder(output)

        # pseeds as percentage
        pseeds = self.Gseeds.graph['pseeds']
        pseeds = int(pseeds * 100) if pseeds <= 1.0 else pseeds

        # 1. graph (with ci, xi, and seed info)
        fn = get_graph_filename(output, pseeds, postfix)
        write_gpickle(self.G, fn)

        # 2. sample graph (nodes, edges, node attributes)
        fn = get_samplegraph_filename(output, pseeds, postfix)
        write_gpickle(self.Gseeds, fn)

        # evaluation
        obj = {}
        obj['N'] = self.G.graph['N']
        obj['m'] = self.G.graph['m']
        obj['B'] = self.G.graph['B']
        obj['H'] = self.G.graph['H']
        obj['i'] = self.G.graph['i']
        obj['x'] = self.G.graph['x']

        obj['sampling'] = self.Gseeds.graph['method']
        obj['pseeds'] = self.Gseeds.graph['pseeds']
        obj['epoch'] = self.Gseeds.graph['epoch']

        obj['n'] = self.G.number_of_nodes()
        obj['e'] = self.G.number_of_edges()
        obj['min_degree'] = self.G.graph['min_degree']

        obj['rocauc'] = self.rocauc
        obj['mae'] = self.mae
        obj['ccm'] = self.ccm
        obj['ccM'] = self.ccM
        obj['bias'] = self.bias
        obj['lag'] = self.duration

        # 3. dictionary with evaluation metrics (json file)
        fn = get_evaluation_filename(output, pseeds, postfix)
        write_pickle(obj, fn)

        # going back to default current loc
        os.chdir(tmp)

    @staticmethod
    def update_all_results(path, kind, sampling="all", njobs=1, verbose=True):
        s = sampling if sampling != "all" else ""
        k = kind if kind != "all" else ""

        files = [os.path.join(path, folder, fn) for folder in os.listdir(path)
                 for fn in os.listdir(os.path.join(path, folder))
                 if os.path.isdir(os.path.join(path, folder)) and folder.endswith(s)
                 and folder.startswith(k) and fn.endswith(".pickle") and "evaluation" in fn]

        _ = Parallel(n_jobs=njobs)(delayed(_update_pickle_to_dataframe)(fn, verbose) for fn in files)
        return

    @staticmethod
    def get_all_results_as_dataframe(path, kind, sampling="all", njobs=1, verbose=True):
        s = sampling if sampling != "all" else ""
        k = kind if kind != "all" else ""

        files = [os.path.join(path, folder, fn) for folder in os.listdir(path)
                 for fn in os.listdir(os.path.join(path, folder))
                 if os.path.isdir(os.path.join(path, folder)) and folder.endswith(s)
                 and folder.startswith(k) and fn.endswith(".pickle") and "evaluation" in fn]

        results = Parallel(n_jobs=njobs)(delayed(_load_pickle_to_dataframe)(fn, verbose) for fn in files)
        df = pd.concat(results).reset_index(drop=True)
        df.loc[:, 'network_size'] = df.apply(lambda row: "N{}, m{}".format(row["N"], row["m"]), axis=1)
        return df

    @staticmethod
    def get_graph_filenames(path):
        return [os.path.join(path, fn) for fn in os.listdir(path) if fn.endswith(".gpickle") and fn.startswith("P") and "_graph" in fn]

    @staticmethod
    def get_samplegraph_filenames(path):
        return [os.path.join(path, fn) for fn in os.listdir(path) if fn.endswith(".gpickle") and fn.startswith("P") and "_samplegraph" in fn]

    @staticmethod
    def get_evaluation_filenames(path):
        return [os.path.join(path, fn) for fn in os.listdir(path) if fn.endswith(".pickle") and fn.startswith("P") and "_evaluation" in fn]