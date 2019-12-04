from collections import Counter

import numpy as np
import pandas as pd


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
    counts = Counter([ '{}{}'.format(graph.graph['group'][graph.graph['labels'].index(graph.node[edge[0]][graph.graph['class']])], graph.graph['group'][graph.graph['labels'].index(graph.node[edge[1]][graph.graph['class']])]) for edge in graph.edges()])
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

def get_degrees(graph):
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

    # normalize
    condprob = condprob.div(condprob.sum(axis=1), axis=0)

    return condprob
