import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random as rd
from scipy.optimize import fsolve
from collections import Counter 
from joblib import Parallel 
from joblib import delayed
import pandas as pd
import os
import powerlaw

from generate_homophilic_graph_asymmetric import homophilic_barabasi_albert_graph_assym

####################################################################
# Fariba's code
####################################################################

#################################
# This function is useful if
# you have an empirical in-group edges
# and want to calculate numerically
# the values of homophily.
################################

def numeric_solver(p):
    h_aa,h_bb,ca,beta_a,beta_b = p
    
    fb = 1 - minority_fraction
    fa = minority_fraction

    M = maj_maj + maj_min + min_min
    m_bb = maj_maj
    m_ab = maj_min
    m_aa = min_min

    pbb = float(m_bb)/(m_aa+m_bb+m_ab)
    paa = float(m_aa)/(m_aa+m_bb+m_ab)
    pba = float(m_ab)/(m_aa+m_bb+m_ab)
    pab = pba


    h_ab = 1- h_aa
    h_ba = 1- h_bb

    A = (h_aa - h_ab)*(h_ba - h_bb)
    B = ((2*h_bb - (1-fa) * h_ba)*(h_aa - h_ab) + (2*h_ab - fa*(2*h_aa - h_ab))*(h_ba - h_bb))
    C = (2*h_bb*(2*h_ab - fa*(2*h_aa - h_ab)) - 2*fa*h_ab*(h_ba - h_bb) - 2*(1-fa)*h_ba * h_ab)
    D = - 4*fa*h_ab*h_bb
    
    P = [A, B, C, D]

    Z = fa / (1 - beta_a)
    K = fb / (1 - beta_b)

    #p_aa = (fa * h_aa *ba)/( h_aa *ba + h_ab *bb) # this is the exact result
    #p_bb = (fb * h_bb *bb)/( h_bb *bb + h_ba *ba) # this is the exact result    
    #p_ab = (fa*h_ab*bb) /(h_ab*bb +h_aa*ba) + (fb*h_ba*ba)/(h_ba*ba +h_bb*bb)
    #P_aa_analytic = float(p_aa)/(p_aa + p_bb + p_ab) # this is the exact result
    #P_bb_analytic = float(p_bb)/(p_aa + p_bb + p_ab) # this is the exact result

    return ((pbb* ((fa * h_aa * Z)+ (fb*(1-h_bb)*Z) + (fa*(1-h_aa)*K)+(fb *h_bb*K) )) - (fb * h_bb * K )  , 
            
            (paa* ((fa * h_aa * Z)+ (fb*(1-h_bb)*Z) + (fa*(1-h_aa)*K)+(fb *h_bb*K) )) - (fa * h_aa * Z )  ,
            
            beta_a - float(fa*h_aa)/ (h_aa*ca + h_ab * (2 - ca)) - float(fb*h_ba)/(h_ba*ca + h_bb*(2-ca)) ,
            
            beta_b - float(fb*h_bb)/ (h_ba*ca + h_bb * (2 - ca)) - float(fa*h_ab)/(h_aa*ca + h_ab*(2-ca)) , 
            
            ca - np.roots(P)[root_num] )
    

    
def measuring_empirical_homophily(maj_maj,maj_min,min_min,minority_fraction):
    
    '''
    the estimation is largely depends on the initial values that are fed to the fsolver.
    therefore it is important to initialize the fsolve functions with realist values to begin with.
    Note that the exponent of the degree distribution (e) is related to the degree growth (beta)
    as follows: e = float(1)/beta + 1. Therefore, if one can estimate the exponent of the 
    degree distribution for the minority and majority, it will help to initialze the values for beta.
    
    maj-maj etc are total number of majority to majority links. etc

    '''

    global root_num

    for root_num in [0,1,2]:
        
        # h_aa,h_bb,ca,beta_a,beta_b
        # these are initializations that you need to tune
        
        h_aa_anal,h_bb_anal,ca,beta_a,beta_b = fsolve(numeric_solver,(1,1,0.5,0.5,0.5)) 
        
        print('estimations',h_aa_anal,h_bb_anal,ca, beta_a , beta_b)
        e_min,e_maj = float(1)/beta_a + 1, float(1)/beta_b + 1
        
        if 0<ca and ca<2: #this is acceptable range for ca
            return h_aa_anal, h_bb_anal , e_min , e_maj 
        
        
####################################################################
# Lisette's code
####################################################################

def load_graph(fn):
    return nx.read_gpickle(fn)

def get_minority_fraction(G):
    tmp = Counter([G.graph['group'][G.graph['labels'].index(G.node[n][G.graph['class']])] for n in G.nodes()])
    return tmp['m'] / float(tmp['m']+tmp['M'])


def get_edge_type_counts(G, fraction=False):
    tmp = Counter(['{}{}'.format( G.graph['group'][G.graph['labels'].index(G.node[e[0]][G.graph['class']])], G.graph['group'][G.graph['labels'].index(G.node[e[1]][G.graph['class']])] ) for e in G.edges()])
    min_min , maj_min , min_maj, maj_maj = tmp['mm'], tmp['Mm'] , tmp['mM'], tmp['MM']
    
    if fraction:
        total = min_min + maj_min + min_maj + maj_maj
        min_min , maj_min , min_maj, maj_maj =  min_min/total , maj_min/total , min_maj/total , maj_maj/total
        
    return min_min , maj_min , min_maj, maj_maj

##########################################################################
# Given a specific empirical network
##########################################################################

def _get_diff(hmm,hMM, N,m, fm, Emm, EMm, EmM, EMM):
    hmm = round(hmm,2)
    hMM = round(hMM,2)
    g = homophilic_barabasi_albert_graph_assym(N, m, fm, round(1.0-hmm,2), round(1.0-hMM,2))
    emm, eMm, emM, eMM = get_edge_type_counts(g,True) 
    diff = abs(Emm-emm)+abs(EMM-eMM)+abs( (EmM+EMm) - (emM+eMm) )
    print('hmm:{:.2f} hMM:{:.2f} diff:{:.10f}'.format(hmm,hMM,diff))
    del(g)
    return (diff, (hmm,hMM))

def get_homophily_MLE(G, njobs=1):
    N = 1000
    m = 2
    fm = get_minority_fraction(G)
    Emm, EMm, EmM, EMM = get_edge_type_counts(G, True)
    print('original: ',Emm,EMm,EmM,EMM)
    
    results_diff = []
    results_h = []
    
    h = np.arange(0.00, 1.05, 0.05)
    results = Parallel(n_jobs=njobs)(delayed(_get_diff)(hmm,hMM,N,m,fm,Emm, EMm, EmM, EMM) for hmm in h for hMM in h) 
    results_diff = [r[0] for r in results]
    results_h = [r[1] for r in results]

    best = np.argmin(results_diff)
    return results_h[best]
            
##########################################################################
# For all fm and hmm, hMM
##########################################################################

def _get_evalues(N, m, fm, hmm, hMM):
    hmm = round(hmm,2)
    hMM = round(hMM,2)
    g = homophilic_barabasi_albert_graph_assym(N, m, fm, round(1.0-hmm,2), round(1.0-hMM,2))
    emm, eMm, emM, eMM = get_edge_type_counts(g,True) 
    del(g)
    return pd.DataFrame({'N':N,
                         'm':m,
                         'fm':fm,
                         'hmm':hmm,
                         'hMM':hMM,
                         'emm':emm,
                         'eMM':eMM,
                         'emM':emM,
                         'eMm':eMm,
                        }, index=[0], columns=['N','m','fm','hmm','hMM','emm','eMM','emM','eMm'])

def generate_E_values(fms, njobs=1, fn=None):
    
    if os.path.exists(fn):
        print('{} loading...'.format(fn))
        df = pd.read_csv(fn, index_col=False)
        return df
    
    fms = list(fms)
    N = 1000
    m = 2
    h = np.arange(0.00, 1.01, 0.01)
    results = Parallel(n_jobs=njobs)(delayed(_get_evalues)(N, m, fm, hmm, hMM) for hmm in h for hMM in h for fm in fms) 
    print('{} results.'.format(len(results)))
    
    df = pd.concat(results, ignore_index=True)
    if fn is not None:
        df.to_csv(fn, index=False)
        print('{} saved!'.format(fn))
        
    return df

def find_homophily_MLE(G, df):
    fm = round(get_minority_fraction(G),2)
    Emm, EMm, EmM, EMM = get_edge_type_counts(G, True)
    
    tmp = df.query("fm==@fm").copy()
    tmp.loc[:,'diff'] = tmp.apply(lambda row: abs(Emm-row.emm)+abs((EMm+EmM)-(row.eMm+row.emM))+abs(EMM-row.eMM), axis=1)
    id = tmp['diff'].idxmin()
    
    return df.loc[id,'hmm'], df.loc[id,'hMM']
      
    
##########################################################################
# Out-Degree Distribution
##########################################################################

def fit_power_law(data, discrete=True):
    return powerlaw.Fit(data,
                        discrete=discrete,
                        verbose=False)

def get_outdegree_powerlaw_exponents(g):
         
    x = np.array([d for n, d in g.degree() if g.graph['labels'].index(g.node[n][g.graph['class']]) == 0])
    fitM = fit_power_law(x)

    x = np.array([d for n, d in g.degree() if g.graph['labels'].index(g.node[n][g.graph['class']]) == 1])
    fitm = fit_power_law(x)

    return fitm, fitM