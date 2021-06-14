import glob
from collections import Counter

import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from utils import io


##################################################################################
# Networks
##################################################################################

def get_param(datafn, key):
    # BAH-N2000-m20-B0.1-H0.0-i1-x5-h0.0-k37.9-km189.4-kM21.0.gpickle
    # Swarthmore42.gpickle
    # BAH-Swarthmore42-N1519-m2-B0.49-Hmm0.54-HMM0.51-i1-x5-h0.5-k4.0-km4.1-kM3.9_nodes

    if not key.startswith("-"):
        key = '-{}'.format(key)

    if key in datafn:
        tokens = datafn.split(key)

        if key == '-H' and len(tokens) == 3 and '-Hmm' in datafn and '-HMM' in datafn:
            Hmm = float(datafn.split('-Hmm')[-1].split('-')[0])
            HMM = float(datafn.split('-HMM')[-1].split('-')[0])
            return round(np.mean([Hmm,HMM]),2)

        elif key != '-H':
            return tokens[-1].split("-")[0].replace(".gpickle","")

    return None

def get_edge_type_counts(graph, fractions=False):
    counts = Counter([ '{}{}'.format(graph.graph['group'][graph.graph['labels'].index(graph.node[edge[0]][graph.graph['class']])],
                                     graph.graph['group'][graph.graph['labels'].index(graph.node[edge[1]][graph.graph['class']])]) for edge in graph.edges()])
    if fractions:
        total = float(counts['mm'] + counts['mM'] + counts['MM'] + counts['Mm'])
        if total == 0:
            return 0, 0, 0, 0
        return counts['mm']/total, counts['mM']/total, counts['MM']/total, counts['Mm']/total

    return counts['mm'], counts['mM'], counts['MM'], counts['Mm']

def get_homophily(graph, smooth=1):

    fm = get_minority_fraction(graph)
    Emm, EmM, EMM, EMm = get_edge_type_counts(graph)

    Emm += smooth
    EmM += smooth
    EMM += smooth
    EMm += smooth

    fM = 1 - fm
    emm = float(Emm) / (Emm + EmM + EMm + EMM)
    return float(-2 * emm * fm * fM) / ((emm * (fm ** 2)) - (2 * emm * fm * fM) + (emm * (fM ** 2) - (fm ** 2)))

def get_similitude(graph):
    if graph.number_of_edges() == 0:
        return 0.5

    h = round(sum([int(graph.node[edge[0]][graph.graph['class']]==graph.node[edge[1]][graph.graph['class']]) for edge in graph.edges()]) / graph.number_of_edges(),1)
    return h

def get_minority_fraction(graph):
    b = Counter([graph.node[n][graph.graph['class']] for n in graph.nodes()]).most_common()[1][1] / graph.number_of_nodes()
    return b

def get_average_degrees(graph):
    labels = graph.graph['labels']
    group = graph.graph['group']

    k = []
    km = []
    kM = []

    for n in graph.nodes():
        k.append(graph.degree(n))

        if group[labels.index(graph.node[n][graph.graph['class']])] == "m":
            km.append(graph.degree(n))
        else:
            kM.append(graph.degree(n))

    return np.mean(k),np.mean(km),np.mean(kM)

def get_min_degree(graph):
    return min([d for n, d in graph.degree()])

def get_density(graph):
    return nx.density(graph)


def get_expected_N_E(density, m=2, E=1000, verbose=False):
    N = np.sqrt(2*E/density)
    enough = E >= m*N

    if not enough:
        N = ((2*m)/density)+1
        
    N = int(np.ceil(N))
    E = (N*(N-1)/2)*density
    E = int(np.ceil(E))
    
    if verbose:
        print(N,E)
        
    return N, E


def _get_global_estimates(index, row, cols, output):

    # BAH-N2000-m20-B0.3-H0.9-i3-x5-h0.9-k39.6-km36.5-kM40.9_nodes 11
    # Caltech36_nodes 1
    # BAH-Caltech36-N701-m2-B0.33-H0.0-i1-x5-h0.5-k4.0-km5.0-kM3.5_nodes 12
    # BAH-Caltech36-N701-m2-B0.33-Hmm0.63-HMM0.44-i1-x5-h0.5-k4.0-km5.0-kM3.5_nodes 13

    if row['kind'] == 'empirical':
        # empirical
        files = glob.glob(output + '/{}_*/P50_graph_1.gpickle'.format(row['dataset']), recursive=True)
    elif row['dataset'] == '-':
        # model
        files = glob.glob(output + '/{}*N{}*m{}*B{}*H{}*/P50_graph_1.gpickle'.format(row['kind'], row['N'], row['m'], row['B'],row['H']), recursive=True)
    else:
        # fit assymetric
        files = glob.glob(output + '/{}-{}-N{}-m{}-B{}-Hmm{}-HMM{}-*/P50_graph_1.gpickle'.format(row['kind'], row['dataset'],row['N'],row['m'],row['B'],row['Hmm'],row['HMM']), recursive=True)
        
        # fit symmetric
        if len(files) == 0:
            files = glob.glob(output + '/{}-{}-N{}-m{}-B{}-H{}-*/P50_graph_1.gpickle'.format(row['kind'], row['dataset'],row['N'],row['m'],row['B'],row['H']), recursive=True)
            
    if len(files) <= 0:
        raise Exception('network does not exist: {} {}\n{}'.format(index, len(files), row))

    p, cp = None, None
    for fn in files:
        G = io.load_gpickle(fn)
        d = get_density(G)

        if d == row['density']:
            p = prior_learn(G)
            cp = nBC_learn(G)
            break

    if p is None or cp is None:
        print()
        print(len(files))
        print(files)
        print(G.graph)
        print(d, row['density'])

        raise Exception('network does not exist with density: \n{}'.format(row))

    # {'attributes': ['color'],
    #  'class': 'color',
    #  'density': 0.019750375187593795,
    #  'fullname': 'BAH-N2000-m20-B0.5-H1.0-i5-x5-h1.0-k39.5-km39.5-kM39.4',
    #  'group': ['M', 'm'],
    #  'kind': 'BAH',
    #  'labels': ['blue', 'red'],
    #  'name': 'homophilic_barabasi_albert'}
    # BAH-N2000-m20-B0.5-H1.0-i5-x5-h1.0-k39.5-km39.5-kM39.4

    # realworld
    # {'attributes': ['gender'],
    #  'class': 'gender',
    #  'density': 0.0630283268799674,
    #  'group': ['M', 'm'],
    #  'labels': [2, 1],
    #  'name': 'Caltech36'}

    return pd.DataFrame({'kind': row['kind'],
                         'dataset': row['dataset'],
                         'N': row['N'],
                         'm': row['m'],
                         'density': d,
                         'B': row['B'],
                         'H': row['H'],
                         'Hmm': row['Hmm'] if 'Hmm' in row else None,
                         'HMM': row['HMM'] if 'HMM' in row else None,
                         'p0': p.iloc[0],
                         'p1': p.iloc[1],
                         'cp00': cp.iloc[0, 0],
                         'cp01': cp.iloc[0, 1],
                         'cp10': cp.iloc[1, 0],
                         'cp11': cp.iloc[1, 1],
                        }, columns=cols, index=[0])

def get_global_estimates(df, LC, RC, output, njobs=1):
    from org.gesis.local.local import CLASS_PRIOR
    from org.gesis.relational.relational import NETWORK_ONLY_BAYES
    from org.gesis.relational.relational import NETWORK_ONLY_LINK_BASED

    if LC != CLASS_PRIOR:
        raise Exception("LC {} does not exist.".format(LC))

    if RC not in [NETWORK_ONLY_BAYES,NETWORK_ONLY_LINK_BASED]:
        raise Exception("RC {} does not exist.".format(RC))

    cols = ['kind','dataset','N','m','density','B','H','Hmm','HMM','p0','p1','cp00','cp01','cp10','cp11']
    results = Parallel(n_jobs=njobs)(delayed(_get_global_estimates)(index,row,cols,output) for index, row in df.iterrows())

    return pd.concat(results, ignore_index=True)

def merge_global_estimates(df, LC, RC, output, njobs=1):
    gcols = ['kind', 'dataset', 'N', 'm', 'B', 'H', 'Hmm', 'HMM', 'density']

    df_target = df.groupby(gcols).size().reset_index()
    #print(df_target.head(1))
    print(df_target.shape)
    print(df_target[gcols].sample(10))

    df_global_estimates = get_global_estimates(df_target, LC, RC, output, njobs)
    #print(df_global_estimates.head(1))
    print(df_global_estimates.shape)
    print(df_global_estimates[gcols].sample(10))

    df = df.merge(df_global_estimates, left_on=gcols, right_on=gcols, how='left', suffixes=("", "_g"))
    print(df.head(1))
    print(df.shape)

    df.loc[:, 'EEp0'] = df.apply(lambda row: row['p0'] - row["p0_g"], axis=1)
    df.loc[:, 'EEp1'] = df.apply(lambda row: row['p1'] - row["p1_g"], axis=1)
    df.loc[:, 'EEcp00'] = df.apply(lambda row: row['cp00'] - row["cp00_g"], axis=1)
    df.loc[:, 'EEcp10'] = df.apply(lambda row: row['cp10'] - row["cp10_g"], axis=1)
    df.loc[:, 'EEcp11'] = df.apply(lambda row: row['cp11'] - row["cp11_g"], axis=1)
    df.loc[:, 'EEcp01'] = df.apply(lambda row: row['cp01'] - row["cp01_g"], axis=1)

    df.loc[:, 'SEp0'] = df.apply(lambda row: (row['p0'] - row["p0_g"]) ** 2, axis=1)
    df.loc[:, 'SEp1'] = df.apply(lambda row: (row['p1'] - row["p1_g"]) ** 2, axis=1)
    df.loc[:, 'SEcp00'] = df.apply(lambda row: (row['cp00'] - row["cp00_g"]) ** 2, axis=1)
    df.loc[:, 'SEcp11'] = df.apply(lambda row: (row['cp11'] - row["cp11_g"]) ** 2, axis=1)

    df.loc[:, 'SEcpDiff'] = df.apply(lambda row: row['SEcp00'] - row['SEcp11'], axis=1)
    df.loc[:, 'SEcpSum'] = df.apply(lambda row: row['SEcp00'] + row['SEcp11'], axis=1)
    df.loc[:, 'SE'] = df.apply(lambda row: row['SEp1'] + row['SEcp00'] + row['SEcp11'], axis=1)
    return df

##################################################################################
# Collective Relational Classification
##################################################################################

def prior_learn(Gseeds):
    tmp = Counter([Gseeds.node[n][Gseeds.graph['class']] for n in Gseeds.nodes()])

    prior = pd.Series(index=Gseeds.graph['labels'])
    prior.loc[Gseeds.graph['labels'][0]] = tmp[Gseeds.graph['labels'][0]] + 1
    prior.loc[Gseeds.graph['labels'][1]] = tmp[Gseeds.graph['labels'][1]] + 1

    prior /= prior.sum()

    return prior

def nBC_learn(Gseeds, smoothing=True, weight=None):
    # measuring probabilities based on Bayes theorem
    condprob = pd.DataFrame(index=Gseeds.graph['labels'],
                                 columns=Gseeds.graph['labels'],
                                 data=np.zeros((len(Gseeds.graph['labels']), len(Gseeds.graph['labels']))))

    # counting edges (faster than traversing nodes^2)
    edge_counts = Counter([(Gseeds.node[edge[0]][Gseeds.graph['class']],
                            Gseeds.node[edge[1]][Gseeds.graph['class']]) for edge in Gseeds.edges()])

    # exact edge counts
    for k, v in edge_counts.items():
        s, t = k
        condprob.loc[s, t] = v

    # if undirected correct for same-class links (times 2), different-class link both ways
    if not Gseeds.is_directed():
        labels = Gseeds.graph['labels']
        condprob.loc[labels[0], labels[0]] *= 2
        condprob.loc[labels[1], labels[1]] *= 2
        tmp = condprob.loc[labels[0], labels[1]] + condprob.loc[labels[1], labels[0]]
        condprob.loc[labels[0], labels[1]] = tmp
        condprob.loc[labels[1], labels[0]] = tmp

    # Laplace smoothing
    if smoothing:
        condprob += 1

    # normalize (row-wise)
    condprob = condprob.div(condprob.sum(axis=1), axis=0)

    return condprob

##################################################################################
# Analytical Probabilities / Estiamtion / Error
# Beta values and more info on derivations:
# Karimi, F., Génois, M., Wagner, C., Singer, P., & Strohmaier, M. (2018).
# Homophily influences ranking of minorities in social networks.
# Scientific reports, 8(1), 11077.
# https://www.nature.com/articles/s41598-018-29405-7/
##################################################################################

def get_analytical_probabilities(B, H):
    #        B1    B:   H
    betas = {0.5: {0.2: {'b1': 0.5, 'b0': 0.5},
                   0.5: {'b1': 0.5, 'b0': 0.5},
                   0.8: {'b1': 0.5, 'b0': 0.5}},
             0.3: {0.2: {'b1': 0.68, 'b0': 0.38},
                   0.5: {'b1': 0.5, 'b0': 0.5},
                   0.8: {'b1': 0.45, 'b0': 0.51}},
             0.1: {0.2: {'b1': 0.88, 'b0': 0.28},
                   0.5: {'b1': 0.5, 'b0': 0.5},
                   0.8: {'b1': 0.28, 'b0': 0.51}}}

    beta_0 = betas[B][H]['b0']
    beta_1 = betas[B][H]['b1']

    d0 = (1 - B) * (1 - B) * H * (1 - beta_1)
    d1 = B * B * H * (1 - beta_0)
    d01 = (1 - B) * B * (1 - H) * ((1 - beta_1) + (1 - beta_0))

    P11 = d1 / (d1 + d0 + d01)
    P00 = d0 / (d1 + d0 + d01)
    P01 = d01 / (d1 + d0 + d01)

    return P00, P11, P01


def get_analytical_estimators(B, H):
    #        B1    B:   H
    betas = {0.5: {0.2: {'b1': 0.5, 'b0': 0.5},
                   0.5: {'b1': 0.5, 'b0': 0.5},
                   0.8: {'b1': 0.5, 'b0': 0.5}},
             0.3: {0.2: {'b1': 0.68, 'b0': 0.38},
                   0.5: {'b1': 0.5, 'b0': 0.5},
                   0.8: {'b1': 0.45, 'b0': 0.51}},
             0.1: {0.2: {'b1': 0.88, 'b0': 0.28},
                   0.5: {'b1': 0.5, 'b0': 0.5},
                   0.8: {'b1': 0.28, 'b0': 0.51}}}

    beta_0 = betas[B][H]['b0']
    beta_1 = betas[B][H]['b1']

    d0 = (1 - B) * (1 - B) * H * (1 - beta_1)
    d1 = B * B * H * (1 - beta_0)
    d01 = (1 - B) * B * (1 - H) * ((1 - beta_1) + (1 - beta_0))

    P11 = d1 / (d1 + d01)
    P01 = d01 / (d1 + d01)

    P00 = d0 / (d0 + d01)
    P10 = d01 / (d0 + d01)

    return P00, P10, P11, P01

def get_small_sample_error(P):
    return np.sqrt(P)


def estimate_homophily_BAH_empirical(graph, fm=None, EMM=None, EMm=None, EmM=None, Emm=None, gammaM_in=None, gammam_in=None, verbose=False):

    hmm = []
    hMM = []
    diff = []

    if graph is not None and (fm is None or EMM is None or EMm is None or EmM is None or Emm is None):
        Emm, EmM, EMM, EMm = get_edge_type_counts(graph)
        fm = get_minority_fraction(graph)
    elif graph is None and (fm is None or EMM is None or EMm is None or EmM is None or Emm is None):
        raise Exception('Missing important parameters.')

    E = EMM + EMm + EmM + Emm
    min_min = Emm / E
    maj_maj = EMM / E
    min_maj = ( EmM + EMm ) / E
    fM = 1 - fm

    # calculating ca for undirected
    K_m = min_min + min_maj
    K_M = maj_maj + min_maj
    K_all = K_m + K_M

    cm = (K_m) / K_all
    if verbose:
        print(cm)

    for h_mm_ in np.arange(0, 1.01, 0.01):
        for h_MM_ in np.arange(0, 1.01, 0.01):

            h_mm_analytical = h_mm_
            h_MM_analytical = h_MM_

            h_mM_analytical = 1 - h_mm_analytical
            h_Mm_analytical = 1 - h_MM_analytical

            if gammaM_in is None:
                try:
                    gamma_M = float(fM * h_MM_analytical) / ((h_Mm_analytical * cm) + (h_MM_analytical * (2 - cm))) + float(fm * h_mM_analytical) / ((h_mm_analytical * cm) + (h_mM_analytical * (2 - cm)))
                except RuntimeWarning:
                    if verbose:
                        print('break 2')
                    break
            else:
                gamma_M = gammaM_in

            if gammam_in is None:
                try:
                    gamma_m = float(fm * h_mm_analytical) / ((h_mm_analytical * cm) + (h_mM_analytical * (2 - cm))) + float(fM * h_Mm_analytical) / ((h_Mm_analytical * cm) + (h_MM_analytical * (2 - cm)))
                except RuntimeWarning as ex:
                    if verbose:
                        print('break 1')
                    break
            else:
                gamma_m = gammam_in

            K = 1 - gamma_m
            Z = 1 - gamma_M

            if ((fm * h_mm_analytical * Z) + ((1 - fm) * (1 - h_mm_analytical) * K) == 0 or (fM * h_MM_analytical * K) + (fm * (1 - h_MM_analytical) * Z)) == 0:
                break

            pmm_analytical = float(fm * h_mm_analytical * Z) / ((fm * h_mm_analytical * Z) + ((1 - fm) * (1 - h_mm_analytical) * K))
            pMM_analytical = float(fM * h_MM_analytical * K) / ((fM * h_MM_analytical * K) + (fm * (1 - h_MM_analytical) * Z))

            if min_min + min_maj + maj_maj == 0:
                # bipartite
                raise NotImplementedError('This model does not support bipartite networks.')
            else:
                pmm_emp = float(min_min) / (min_min + min_maj)
                pMM_emp = float(maj_maj) / (maj_maj + min_maj)

            _diff = abs(pmm_emp - pmm_analytical) + abs(pMM_emp - pMM_analytical)
            diff.append(_diff)
            hmm.append(h_mm_analytical)
            hMM.append(h_MM_analytical)

            if verbose and _diff < 0.02:
                print()
                print('pmm_emp', pmm_emp, 'pmm_analytical', pmm_analytical)
                print('pMM_emp', pMM_emp, 'pMM_analytical', pMM_analytical)
                print('pMM_diff', abs(pmm_emp - pmm_analytical), 'pMM_diff', abs(pMM_emp - pMM_analytical), 'diff', _diff)
                print('h_mm_analytical', h_mm_analytical, 'h_MM_analytical', h_MM_analytical, 'cm_analytical', cm)

    best = np.argmin(diff)
    return hmm[best], hMM[best]

def find_homophily_MLE(G, df):
    fm = round(get_minority_fraction(G),2)
    Emm, EmM, EMM, EMm = get_edge_type_counts(G, True)

    tmp = df.query("fm==@fm").copy()
    tmp.loc[:,'diff'] = tmp.apply(lambda row: abs(Emm-row.emm)+abs((EMm+EmM)-(row.eMm+row.emM))+abs(EMM-row.eMM), axis=1)
    id = tmp['diff'].idxmin()
    
    return df.loc[id,'hmm'], df.loc[id,'hMM']