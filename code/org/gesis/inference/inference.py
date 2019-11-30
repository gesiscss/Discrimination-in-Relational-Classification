############################################
# System dependencies
############################################
import time
import os
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

############################################
# Local dependencies
############################################
from org.gesis.inference.relaxation import Relaxation
from utils.io import create_folder
from utils.io import write_gpickle
from utils.io import write_pickle

############################################
# Constants
############################################
RELAXATION = "relaxation"

############################################
# Functions
############################################

def is_inference_done(root, datafn, pseeds, postfix):
    output = os.path.join(root, os.path.basename(datafn).replace(".gpickle", ""))
    f1 = get_graph_filename(output, pseeds, postfix)
    f2 = get_evaluation_filename(output, pseeds, postfix)
    return os.path.exists(f1) and os.path.exists(f2)

def get_graph_filename(output, pseeds, postfix):
    if pseeds < 1:
        pseeds = int(round(pseeds*100,1))
    return os.path.join(output,"P{}_graph{}.gpickle".format(pseeds, '_{}'.format(postfix) if postfix is not None else ""))

def get_samplegraph_filename(output, pseeds, postfix):
    if pseeds < 1:
        pseeds = int(round(pseeds*100,1))
    return os.path.join(output,"P{}_samplegraph{}.gpickle".format(pseeds, '_{}'.format(postfix) if postfix is not None else ""))

def get_evaluation_filename(output, pseeds, postfix):
    if pseeds < 1:
        pseeds = int(round(pseeds*100,1))
    return os.path.join(output,"P{}_evaluation{}.pickle".format(pseeds, '_{}'.format(postfix) if postfix is not None else ""))

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
        self.ccm = sum([1 for t,p in zip(*[y_true,y_pred]) if t==p and t==1]) / m
        
        # majority performance
        M = len(y_true) - m
        self.ccM = sum([1 for t,p in zip(*[y_true,y_pred]) if t==p and t==0]) / M    
        
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
        create_folder(root,True)
        try:
            output = self.G.graph['fullname']
        except:
            output = self.G.graph['name']
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
        obj['N'] = self.G.number_of_nodes()
        obj['E'] = self.G.number_of_edges()
        obj['pseeds'] = self.Gseeds.graph['pseeds']

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