############################################
# System dependencies
############################################
import time
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

############################################
# Local dependencies
############################################
from org.gesis.inference.relaxation import Relaxation

############################################
# Constants
############################################
RELAXATION = "relaxation"

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
        print("Prediction in {} seconds".format(self.duration))
        print("ROCAUC: {}".format(self.rocauc))
        print("ROCAUC curve: {}".format(self.rocauc_curve))
        print("MAE: {}".format(self.mae))
        print("ccm: {}".format(self.ccm))
        print("ccM: {}".format(self.ccM))
        print("bias: {}".format(self.bias))
        