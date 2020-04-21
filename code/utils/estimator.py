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

    if not key.startswith("-"):
        key = '-{}'.format(key)

    if key in datafn:
        return datafn.split(key)[-1].split("-")[0].replace(".gpickle","")

    return None

def get_edge_type_counts(graph):
    counts = Counter([ '{}{}'.format(graph.graph['group'][graph.graph['labels'].index(graph.node[edge[0]][graph.graph['class']])],
                                     graph.graph['group'][graph.graph['labels'].index(graph.node[edge[1]][graph.graph['class']])]) for edge in graph.edges()])
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

def _get_global_estimates(index, row, cols, output):

    files = glob.glob(output + '/{}*N{}*m{}*B{}*H{}*/P10_graph_1.gpickle'.format(row['kind'],row['N'],row['m'],row['B'],row['H']), recursive=True)

    if len(files) <= 0:
        files = glob.glob(output + '/{}*/P10_graph_1.gpickle'.format(row['kind']), recursive=True)

    if len(files) <= 0:
        raise Exception('network does not exist: {}\n{}'.format(len(files), row))

    p, cp = None, None
    for fn in files:
        G = io.load_gpickle(fn)
        if get_density(G) == row['density']:
            p = prior_learn(G)
            cp = nBC_learn(G)
            break

    if p is None or cp is None:
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
                        'N': row['N'],
                        'm': row['m'],
                        'density': get_density(G),
                        'B': row['B'],
                        'H': row['H'],
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

    if LC != CLASS_PRIOR:
        raise Exception("LC {} does not exist.".format(LC))

    if RC != NETWORK_ONLY_BAYES:
        raise Exception("RC {} does not exist.".format(RC))

    cols = ['kind','N','m','density','B','H','p0','p1','cp00','cp01','cp10','cp11']
    results = Parallel(n_jobs=njobs)(delayed(_get_global_estimates)(index,row,cols,output) for index, row in df.iterrows())

    return pd.concat(results, ignore_index=True)

def merge_global_estimates(df, LC, RC, output, njobs=1):

    df_target = df.groupby(['kind', 'N', 'm', 'density', 'B', 'H']).size().reset_index()
    df_global_estimates = get_global_estimates(df_target, LC, RC, output, njobs)

    df = df.merge(df_global_estimates, left_on=['N', 'B', 'H', 'density'], right_on=['N', 'B', 'H', 'density'], how='left', suffixes=("", "_g"))

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