import pandas as pd
from collections import Counter
import numpy as np


def get_similitude(graph):
    h = round(sum([int(graph.node[edge[0]][graph.graph['class']]==graph.node[edge[1]][graph.graph['class']]) for edge in graph.edges()]) / graph.number_of_edges(),1)
    return h

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

def prior_learn(Gseeds):
    tmp = Counter([Gseeds.node[n][Gseeds.graph['class']] for n in Gseeds.nodes()])
    minority = tmp.most_common(2)[1]
    majority = tmp.most_common(2)[0]

    prior = pd.Series(index=[majority[0], minority[0]])
    prior.loc[majority[0]] = majority[1] / Gseeds.number_of_nodes()
    prior.loc[minority[0]] = minority[1] / Gseeds.number_of_nodes()

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
