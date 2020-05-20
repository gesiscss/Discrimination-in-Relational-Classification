import os

import networkx as nx
import numpy as np
import pandas as pd
import powerlaw
from joblib import Parallel
from joblib import delayed
from org.gesis.network import homophilic_barabasi_albert_graph_assym
from utils import estimator
from utils import io
from org.gesis.network.network import Network

##########################################################################
# For all fm and hmm, hMM
##########################################################################

def load_evalues(output):
    fn = os.path.join(output, 'synthetic_evalues.csv')
    if not os.path.exists(fn):
        print('{} does not exist.'.format(fn))
    return io.load_csv(fn)

def get_summary_datasets(datapath, df_evalues, output=None):

    fn_summary = None
    if output is not None:
        fn_summary = os.path.join(output, 'summary_datasets.csv')

    if fn_summary is not None and os.path.exists(fn_summary):
        df_details = io.load_csv(fn_summary)
    else:
        columns = ['dataset', 'N', 'm', 'class', 'minority', 'B', 'E', 'density', 'Emm', 'EMM', 'EmM', 'gamma', 'gammam', 'gammaM', 'Hmm', 'HMM']
        df_details = pd.DataFrame(columns=columns)

        files = [os.path.join(datapath, fn) for fn in os.listdir(datapath) if fn.endswith('.gpickle') and not fn.startswith('BAH')]

        for fn in files:

            dataset = fn.split('/')[-1].replace('.gpickle','')
            net = Network()
            net.load(fn, ignoreInt=None if dataset in ['GitHub','Wikipedia','Escorts'] else 0 )
            g = net.G

            dataset = g.graph['name']
            m = estimator.get_min_degree(g)
            B = estimator.get_minority_fraction(g)
            Emm, EMm, EmM, EMM = estimator.get_edge_type_counts(g)
            fit, fitm, fitM = get_degree_powerlaw_exponents(g)

            gamma, xmin, xmax = fit.power_law.alpha, fit.power_law.xmin, fit.power_law.xmax
            gamma_M, xmin_M, xmax_M = fitM.power_law.alpha, fitM.power_law.xmin, fitM.power_law.xmax
            gamma_m, xmin_m, xmax_m = fitm.power_law.alpha, fitm.power_law.xmin, fitm.power_law.xmax

            Hmm, HMM = find_homophily_MLE(g, df_evalues)

            N = g.number_of_nodes()
            E = g.number_of_edges()
            density = E / ((N*(N-1))/(1. if nx.is_directed(g) else 2.))

            # g.graph['N'] = N
            # g.graph['E'] = E
            # g.graph['density'] = density
            # g.graph['B'] = B
            # g.graph['gammam'] = gamma_m
            # g.graph['gammaM'] = gamma_M
            # g.graph['Hmm'] = Hmm
            # g.graph['HMM'] = HMM
            # io.write_gpickle(g, fn)

            df_details = df_details.append(pd.DataFrame({'dataset': dataset,
                                                         'N': N,
                                                         'm': m,
                                                         'class': g.graph['class'],
                                                         'minority': g.graph['labels'][1],
                                                         'B': B,
                                                         'E': E,
                                                         'density': density,
                                                         'Emm': Emm,
                                                         'EMM': EMM,
                                                         'EmM': EmM + EMm,
                                                         'gamma': gamma,
                                                         'gammam': gamma_m,
                                                         'gammaM': gamma_M,
                                                         'Hmm': Hmm,
                                                         'HMM': HMM,
                                                         },
                                                        columns=columns,
                                                        index=[0]), ignore_index=True)

        df_details.sort_values('dataset', inplace=True, ascending=True)
        if output is not None:
            io.write_csv(df_details, fn_summary)

            # to latex:
            content = df_details.to_latex(index=False, float_format=lambda x: '%.2f' % x)
            io.write_text(content, fn_summary.replace(".csv", ".tex"))

            # to latex pivot:
            df_pivot = df_details.pivot_table(columns='dataset', aggfunc=np.unique)
            df_pivot = df_pivot.reindex(columns[1:])
            content = df_pivot.to_latex(float_format=lambda x: '%.2f' % x)
            io.write_text(content, fn_summary.replace(".csv", "_pivot.tex"))

    return df_details


##########################################################################
# For all fm and hmm, hMM
##########################################################################

def _get_evalues(N, m, fm, hmm, hMM):
    hmm = round(hmm, 2)
    hMM = round(hMM, 2)
    g = homophilic_barabasi_albert_graph_assym(N, m, fm, round(1.0 - hmm, 2), round(1.0 - hMM, 2))
    emm, emM, eMM, eMm = estimator.get_edge_type_counts(g, True)
    del (g)
    return pd.DataFrame({'N': N,
                         'm': m,
                         'fm': fm,
                         'hmm': hmm,
                         'hMM': hMM,
                         'emm': emm,
                         'eMM': eMM,
                         'emM': emM,
                         'eMm': eMm,
                         }, index=[0], columns=['N', 'm', 'fm', 'hmm', 'hMM', 'emm', 'eMM', 'emM', 'eMm'])

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
    fm = round(estimator.get_minority_fraction(G), 2)
    Emm, EmM, EMM, EMm = estimator.get_edge_type_counts(G, True)

    tmp = df.query("fm==@fm").copy()
    tmp.loc[:, 'diff'] = tmp.apply(
        lambda row: abs(Emm - row.emm) + abs((EMm + EmM) - (row.eMm + row.emM)) + abs(EMM - row.eMM), axis=1)
    id = tmp['diff'].idxmin()

    return df.loc[id, 'hmm'], df.loc[id, 'hMM']


##########################################################################
# Out-Degree Distribution
##########################################################################

def get_power_law(fn, bestk=True):
    g = io.load_gpickle(fn)
    data = [d for n, d in g.degree()]
    return fit_power_law(data, bestk=bestk)


def fit_power_law(data, discrete=True, bestk=True):
    if bestk:
        return powerlaw.Fit(data,
                            discrete=discrete,
                            verbose=False)
    return powerlaw.Fit(data,
                        xmin=min(data),
                        xmax=max(data),
                        discrete=discrete,
                        verbose=False)


def get_degree_powerlaw_exponents(g):
    x = np.array([d for n, d in g.degree()])
    fit = fit_power_law(x)

    x = np.array([d for n, d in g.degree() if g.graph['labels'].index(g.node[n][g.graph['class']]) == 0])
    fitM = fit_power_law(x)

    x = np.array([d for n, d in g.degree() if g.graph['labels'].index(g.node[n][g.graph['class']]) == 1])
    fitm = fit_power_law(x)

    return fit, fitm, fitM