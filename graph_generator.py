import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import matplotlib
matplotlib.use('Agg')

import sys
import os
from random import randint
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
from matplotlib import pyplot, patches
import numpy as np
try:
    from graph_tool.all import *
    from graph_tool.spectral import adjacency
except:
    pass
from itertools import product
try:
    import powerlaw
except:
    pass
from joblib import Parallel, delayed
from collections import Counter
from itertools import combinations_with_replacement
import pickle
from libs.network.generate_homophilic_graph_symmetric import *
import matplotlib.ticker as ticker
import pandas as pd
from matplotlib.ticker import AutoMinorLocator

##################################################
# Constants
##################################################

HOMOPHILIC = 'Homophilic'
HETEROPHILIC = 'Heterophilic'
NEUTRAL = 'Random'
BLOCKMODEL_POISSON = 'SBMPPOISSON'
BLOCKMODEL_POWERLAW = 'SBMPL'
BLOCKMODEL_PPL = 'SMBPPL'
PSEUDO_SBM = 'PSEUDOSBM'
CONFIGURATION_MODEL_POWER_LAW = 'CONFMPL'
CONFIGURATION_MODEL_POISSON = 'CONFMPOISSON'
SBM = 'SBM'
BAHM = 'BAHm'


##################################################
# Generators
##################################################

def run_colored_graphs():
    for type in [HOMOPHILIC,HETEROPHILIC,NEUTRAL]:
        label = 'color'
        G = colored(type)
        path = os.path.join(output, 'colored')
        create_path(path)

        # info
        save(G,type, path)
        plot(G,type,label, path)
        info(G,label)
        draw_adjacency_matrix(G, type, path)

def sbm_old(h, classmembership, skew, label, N, B, prefix, path):

    size_1 = int(round(classmembership * N))
    size_2 = N - size_1
    node_blocks = {v: int(v >= size_1) for v in range(N)}
    degree = {b:None for b in range(B)}
    dd = []

    if prefix == BLOCKMODEL_POISSON:
        dd.append( np.random.poisson(skew[0], size_1).astype(int) )
        dd.append( np.random.poisson(skew[1], size_2).astype(int) )
    elif prefix == BLOCKMODEL_POWERLAW:
        dd.append(powerlaw.Power_Law(xmin=1.0, parameters=[skew[0]], discrete=True).generate_random(N).astype(int))
        dd.append(powerlaw.Power_Law(xmin=1.0, parameters=[skew[1]], discrete=True).generate_random(N).astype(int))

    for block in range(B):
        nodes = [v for v, b in node_blocks.items() if b == block]
        degree[block] = {nodes[index]: degree for index, degree in enumerate(dd[block])}

    print('')
    print('===================================================================================')
    print('')

    print('{}-{}-{}'.format(h,classmembership,skew))
    print('')

    print(degree[0])
    # raw_input('...')

    print(degree[1])
    # raw_input('...')

    # SBM
    g, bm = random_graph(N,
                          lambda v, b: degree[b][v],
                          directed=False,
                          parallel_edges=False,
                          self_loops=False,
                          model="blockmodel",
                          block_membership=lambda v: node_blocks[v],
                          edge_probs=lambda a, b: h if a == b else 1 - h)

    # to networkx
    A = adjacency(g)
    G = nx.from_scipy_sparse_matrix(A)
    nx.set_node_attributes(G, label, {i: b for i, b in enumerate(bm)})
    G.graph['attributes'] = [label]

    # info
    type = '{}_H{}_B{}_S{}'.format(prefix, int(h * 100), int(classmembership * 100), '{}'.format('-'.join([str(s).replace('.','') for s in skew])))
    G.name = type
    save(G, type, path)
    plot(G, type, label, path)
    info(G, label)
    draw_adjacency_matrix(G, type, path)
    graph_draw(g, vertex_fill_color=bm, edge_color="black", output_size=(500, 500), output=os.path.join(path, '{}-network2.png'.format(type)))

def run_stochastic_2block_models(prefix,output):
    path = os.path.join(output, prefix)
    create_path(path)

    N = 1000
    B = 2
    homophily = [0.2,0.5,0.8]       # homophily
    class_balance = [0.2,0.5]       # class balance
    lambdas = [1,5,20]              # degree poisson
    alphas = [1.5,3.0]              # degree powerlaw
    label = 'block'

    if prefix == BLOCKMODEL_POWERLAW:
        skewness = list(product(alphas, repeat=B))
    elif prefix == BLOCKMODEL_POISSON:
        skewness = list(product(lambdas, repeat=B))

    print('skewness: {}'.format(skewness))
    runs = 5
    Parallel(n_jobs=runs)(delayed(sbm)(h,classmembership,skew,label,N,B,prefix,path) for h in homophily for classmembership in class_balance for skew in skewness)

def run_pseudo_sbm(label, prefix, output):
    path = os.path.join(output, prefix)
    create_path(path)

    B = 2
    N = 1000
    plt.close('all')
    alphas = [(1.5, 3.5), (3.5, 1.5), (2.0, 2.0)]
    classbalance = [0.2, 0.5]
    homophily = [0.2, 0.5, 0.8]

    for b in classbalance:

        size_1 = int(round(b * N))
        size_2 = N - size_1
        node_blocks = {v:int(v >= size_1) for v in range(N)}

        for c, a in enumerate(alphas):

            f, axarr = plt.subplots(2, 2, figsize=(15, 10))

            theoretical_distribution_1 = powerlaw.Power_Law(xmin=1., parameters=[a[0]], discrete=True)
            simulated_data_1 = theoretical_distribution_1.generate_random(size_1).astype(int)
            fit_1 = powerlaw.Fit(simulated_data_1)
            print('Fitted 1 | xmin:{} alpha:{}'.format(fit_1.power_law.xmin, fit_1.power_law.alpha))
            powerlaw.plot_pdf(simulated_data_1, axarr[0, 0], linewidth=3)
            fit_1.power_law.plot_pdf(simulated_data_1, axarr[0, 0], linestyle='--', color='r')
            simulated_data_1 = simulated_data_1.tolist()
            x1 = sorted(set(simulated_data_1))
            y1 = [simulated_data_1.count(d) for d in x1]
            axarr[1, 0].loglog(x1, y1)
            axarr[1,0].set_ylim(ymin=0.02)
            axarr[0, 0].set_title('alpha:{} | b:{} | xmin:{} | alpha:{}'.format(a[0], b, fit_1.power_law.xmin, round(fit_1.power_law.alpha, 2)))

            theoretical_distribution_2 = powerlaw.Power_Law(xmin=1., parameters=[a[1]], discrete=True)
            simulated_data_2 = theoretical_distribution_2.generate_random(size_2).astype(int)
            fit_2 = powerlaw.Fit(simulated_data_2)
            print('Fitted 2 | xmin:{} alpha:{}'.format(fit_2.power_law.xmin, fit_2.power_law.alpha))
            powerlaw.plot_pdf(simulated_data_2, axarr[0, 1], linewidth=3)
            fit_2.power_law.plot_pdf(simulated_data_2, axarr[0, 1], linestyle='--', color='r')
            simulated_data_2 = simulated_data_2.tolist()
            x2 = sorted(set(simulated_data_2))
            y2 = [simulated_data_2.count(d) for d in x2]
            axarr[1, 1].loglog(x2, y2)
            axarr[1, 1].set_ylim(ymin=0.02)
            axarr[0, 1].set_title('alpha:{} | b:{} | xmin:{} | alpha:{}'.format(a[1], 1. - b, fit_2.power_law.xmin, round(fit_2.power_law.alpha, 2)))

            plt.suptitle('Powerlaw dist.')
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)

            key = '{}_B{}_A{}-{}'.format(prefix, int(b * 100), a[0], a[1])
            fn = os.path.join(path, '{}.png'.format(key))
            plt.savefig(fn)
            plt.close()

            simulated_data = [simulated_data_1,simulated_data_2]

            for h in homophily:
                ### GRAPH
                key = '{}_H{}_B{}_A{}-{}'.format(prefix, int(h * 100), int(b * 100), a[0], a[1])
                print('========================= {} ========================='.format(key))
                G = nx.Graph()

                ## NODES
                G.add_nodes_from(range(N))
                nx.set_node_attributes(G, label, node_blocks)
                print('nodes blocks:\n{}'.format(Counter([n[1][label] for n in G.nodes(data=True)])))
                G.graph['attributes'] = [label]

                ## GLOBAL EDGES (across groups)


                counter = 0
                previous = 0
                converge = 0
                while(True):

                    nodes = G.nodes(data=True)
                    np.random.shuffle(nodes)

                    for n in nodes:
                        need = simulated_data[n[1][label]][n[0] - (0 if n[1][label]==0 else size_1)] - G.degree(n[0])

                        ### same block
                        deg = G.degree(n[0])
                        need1 = int(round(need * h))
                        need2 = need - need1

                        if h >= 0.5:
                            if need1 > 0:
                                othernodes = [ t[0] for t in G.nodes(data=True) if t[1][label] == n[1][label] and t[0] != n[0] and G.degree(t[0]) < simulated_data[t[1][label]][t[0] - (0 if t[1][label]==0 else size_1)] ]
                                np.random.shuffle(othernodes)
                                newedges = othernodes[:1 if counter < 1 else need1]
                                G.add_edges_from([(n[0],t) for t in newedges])

                            if need2 > 0:
                                othernodes = [ t[0] for t in G.nodes(data=True) if t[1][label] != n[1][label] and G.degree(t[0]) < simulated_data[t[1][label]][t[0] - (0 if t[1][label]==0 else size_1)] ]
                                np.random.shuffle(othernodes)
                                newedges = othernodes[:1 if counter < 1 else need2]
                                G.add_edges_from([(n[0],t) for t in newedges])
                                G.add_edges_from([(t,n[0]) for t in newedges])
                        else:

                            if need2 > 0:
                                othernodes = [t[0] for t in G.nodes(data=True) if
                                              t[1][label] != n[1][label] and G.degree(t[0]) < simulated_data[t[1][label]][t[0] - (0 if t[1][label] == 0 else size_1)]]
                                np.random.shuffle(othernodes)
                                newedges = othernodes[:1 if counter < 1 else need2]
                                G.add_edges_from([(n[0], t) for t in newedges])
                                G.add_edges_from([(t, n[0]) for t in newedges])

                            if need1 > 0:
                                othernodes = [t[0] for t in G.nodes(data=True) if
                                              t[1][label] == n[1][label] and t[0] != n[0] and G.degree(t[0]) < simulated_data[t[1][label]][t[0] - (0 if t[1][label] == 0 else size_1)]]
                                np.random.shuffle(othernodes)
                                newedges = othernodes[:1 if counter < 1 else need1]
                                G.add_edges_from([(n[0], t) for t in newedges])


                        print('n: {} | b:{} | degree:{} | powerlaw:{} | need:{} | need1:{} | need2:{} | new degree:{}'.format(n[0], n[1][label], deg,
                                                                                                                              simulated_data[n[1][label]][n[0] - (0 if n[1][label] == 0 else size_1)], need,
                                                                                                                              need1, need2, G.degree(n[0])))

                    counter += 1
                    print('edges: {}'.format(G.number_of_edges()))
                    print('h: {}'.format(len([1 for edge in G.edges() if G.node[edge[0]][label] == G.node[edge[1]][label]])))
                    print('1-h: {}'.format(len([1 for edge in G.edges() if G.node[edge[0]][label] != G.node[edge[1]][label]])))

                    print('0-0: {}'.format(len([1 for edge in G.edges() if G.node[edge[0]][label] == G.node[edge[1]][label] and G.node[edge[0]][label] == 0 ])))
                    print('1-1: {}'.format(len([1 for edge in G.edges() if G.node[edge[0]][label] == G.node[edge[1]][label] and G.node[edge[0]][label] == 1])))
                    print('0-1: {}'.format(len([1 for edge in G.edges() if G.node[edge[0]][label] != G.node[edge[1]][label] and G.node[edge[0]][label] == 0 and G.node[edge[1]][label] == 1])))
                    print('1-0: {}'.format(len([1 for edge in G.edges() if G.node[edge[0]][label] != G.node[edge[1]][label] and G.node[edge[0]][label] == 1 and G.node[edge[1]][label] == 0])))

                    tmp = 0
                    for block in range(B):
                        print('sum degrees theoretical {}: {}'.format(block,sum(simulated_data[block])))
                        print('sum degrees empirical {}: {}'.format(block,sum([G.degree(n[0]) for n in G.nodes(data=True) if n[1][label]==block])))
                        tmp += sum([G.degree(n[0]) for n in G.nodes(data=True) if n[1][label]==block])

                    if previous != 0:
                        converge += int(previous == tmp)

                    if previous != tmp:
                        converge = 0

                    if converge == 10:
                        #raw_input('this is a break after {} iterations'.format(converge))
                        break

                    previous = int(tmp)


                sg1 = G.subgraph([n[0] for n in G.nodes(data=True) if n[1][label] == 0]).copy()
                print('block0:')
                print(nx.info(sg1))

                sg2 = G.subgraph([n[0] for n in G.nodes(data=True) if n[1][label] == 1]).copy()
                print('block1:')
                print(nx.info(sg2))

                # info
                G.name = key
                save(G, key, path)
                plot(G, key, label, path)
                info(G, label)
                draw_adjacency_matrix(G, key, path)








    return

def get_poisson_sequence(n,lam,max_tries=50):
    tries = 0
    while tries < max_tries:
        seq = np.random.poisson(lam, n)
        if nx.is_valid_degree_sequence(seq):
            return seq
        tries += 1
    raise nx.NetworkXError( \
        "Exceeded max (%d) attempts at a valid sequence." % max_tries)

#########################################
# SBM (new)
#########################################

def summary_sbm(g,bm,label,fn,h=None):

    # network x
    A = adjacency(g)
    G = nx.from_scipy_sparse_matrix(A)
    nx.set_node_attributes(G, label, {i: b for i, b in enumerate(bm)})

    # actual h
    homophily = _get_homophily(G, label)
    degrees = _get_degrees(G, label)

    if h is not None:
        if homophily[1] >= (h+0.03) or homophily[1] <= (h-0.03):
            print('homophily required {}, achieved {}'.format(h,homophily[1]))
            return False

    fn = '{}-h{}-k{}-{}'.format(fn, round(homophily[1], 2), round(degrees[0], 1), round(degrees[1], 1))

    path = os.path.dirname(os.path.abspath(fn))
    name = os.path.basename(fn)
    print(path)
    print(name)

    # graph tools
    if not os.path.exists('{}.p'.format(fn)):
        save_sbm(g, bm, fn)
        graph_draw(g, vertex_fill_color=bm, edge_color="black", output_size=(500, 500), output='{}-network2.png'.format(fn))

        # networkx
        G.graph['attributes'] = [label]
        G.name = name

        info(G, label)
        save(G, name, path)
        plot(G, name, label, path)

        draw_adjacency_matrix(G, name, path)
        plot_degree(G, label, '{}-degree.png'.format(fn))
        plot_population(G, label, '{}-population.png'.format(fn))

    return fn

def create_sbm_graph(prefix,label,output):

    path = os.path.join(output, prefix)
    create_path(path)

    ### INITIAL
    N = 1000
    k = 10

    ### CLASS BALANCE
    for b in [0.9]:

        H = 0.8
        print('==========================')
        print('========== INIT ==========')
        print('==========================')
        print('N{} - B{} - H{} - Pk{}'.format(N, b, H, k))
        counter = 0

        while True:
            fn = os.path.join(path, '{}-N{}-B{}-H{}-PK{}'.format(prefix, N, b, H, k))
            g, bm = read_sbm(fn)

            if g is None or bm is None:
                g, bm = random_graph(N,
                                     lambda v, block: np.random.poisson(k),
                                     directed=False,
                                     parallel_edges=False,
                                     self_loops=False,
                                     model="blockmodel",
                                     block_membership=lambda v: int(v >= N * b),
                                     edge_probs=lambda x, y: H)
            else:
                print('loaded!')

            fn = summary_sbm(g, bm, label, fn, H)

            if fn is not False:
                break
            else:
                counter += 1
                if counter == 20:
                    break

        if counter == 20:
            continue


        ### HOMPHILY
        for h in [1.0]:

            if h == H:
                continue

            print('========== REWIRING ==========')
            print('N{} - B{} - H{} - Pk{}'.format(N, b, h, k))
            counter = 0

            while True:
                fnrewired = os.path.join(path, '{}-N{}-B{}-H{}-Pk{}'.format(prefix, N, b, h, k))
                g, bm = read_sbm(fnrewired)

                if g is None or bm is None:
                    # 1. read original graph
                    g, bm = read_sbm(fn)
                    # 2. rewire edges
                    ret = random_rewire(g,
                                        model="blockmodel",
                                        parallel_edges=False,
                                        self_loops=False,
                                        block_membership=bm,
                                        edge_probs=lambda x, y: (h-0.09) if x == y else 1 - (h-0.09) )
                else:
                    print('loaded!')
                flag = summary_sbm(g, bm, label, fnrewired, h)

                if flag is not False:
                    break
                else:
                    counter += 1

                    if counter == 20:
                        break

            if counter == 20:
                continue

    return


#########################################
# SBM AND CNF
#########################################

def get_colors(size):
    return iter(cm.rainbow(np.linspace(0,1,size)))

def plot_degree(G,label,fn):
    font = {'family': 'serif',
            'color': 'darkred',
            'weight': 'normal',
            'size': 8,
            }
    labels = sorted(list(set([G.node[n][label] for n in G])))
    labels.append(None)
    colors = get_colors(len(labels))
    degrees = dict(nx.degree(G))
    tmp = []

    for l in labels:
        if l is None:
            degree_sequence = sorted(degrees.values(), reverse=True)  # degree sequence
        else:
            degree_sequence = sorted([d for n,d in degrees.items() if G.node[n][label]==l],reverse=True)
        k = np.mean(degree_sequence)
        x = sorted(set(degree_sequence))
        y = [degree_sequence.count(d) for d in x]
        plt.loglog(x, y, color=next(colors), marker='o', label='{}'.format('All nodes' if l is None else 'Label {} (k={})'.format(l,round(k,2))))
        tmp.extend(x)
    tmp = min(tmp)
    plt.title("Degree Distribution")
    plt.ylabel("frequency")
    plt.xlabel("degree")
    plt.legend()
    plt.text(tmp+1., 10., r'{}'.format(nx.info(G)), fontdict=font)
    plt.savefig(fn)
    plt.close()

def plot_population(G, label, fn):
    data = [G.node[n][label] for n in G.nodes()]
    counts = Counter(data)

    x = np.arange(len(counts))
    labels = sorted(counts.keys())
    y = [counts[i] for i in labels]

    plt.bar(x,y)
    plt.xticks(x,labels)
    plt.xlabel('class label')
    plt.ylabel('frequency')
    plt.title(label.upper())
    plt.tight_layout()
    plt.savefig(fn)
    plt.close()

def _add_edges(G,label,nodes,h):
    for source in nodes:
        homophily = _get_homophily(G, label)

        if h - homophily[1] > 0:
            targets = [n for n in G.nodes() if G.node[n][label] == G.node[source][label] and n != source]
            np.random.shuffle(targets)
            while len(targets) > 0:
                target = targets.pop()
                if not G.has_edge(source, target):
                    G.add_edge(source, target)
                    break

        elif h - homophily[1] < 0:
            targets = [n for n in G.nodes() if G.node[n][label] != G.node[source][label] and n != source]
            np.random.shuffle(targets)
            while len(targets) > 0:
                target = targets.pop()
                if not G.has_edge(source, target):
                    G.add_edge(source, target)
                    break

        else:
            targets = [n for n in G.nodes() if G.node[n][label] == G.node[source][label] and n != source]
            np.random.shuffle(targets)
            while len(targets) > 0:
                target = targets.pop()
                if not G.has_edge(source, target):
                    G.add_edge(source, target)
                    break

            targets = [n for n in G.nodes() if G.node[n][label] != G.node[source][label] and n != source]
            np.random.shuffle(targets)
            while len(targets) > 0:
                target = targets.pop()
                if not G.has_edge(source, target):
                    G.add_edge(source, target)
                    break
    return G

def sbm(h,b,degree,label,N,C,prefix,path):

    name = '{}-B{}-H{}'.format(prefix, b, h)
    files = [f for f in os.listdir(path) if f.startswith(name)]

    if len(files) == 6:
        print('B{} | H{} | already exists!'.format(b,h))
        return
    else:
        for f in files:
            fn = os.path.join(path,f)
            print('to delete: {}'.format(fn))
            os.remove(fn)

    size_1 = int(round(b * N))
    size_2 = N - size_1

    # SBM
    g, bm = random_graph(N,
                         lambda v, block: np.random.poisson(degree),
                         directed=False,
                         parallel_edges=False,
                         self_loops=False,
                         model="blockmodel",
                         block_membership=lambda v: int(v >= size_1),
                         edge_probs=lambda x, y: h if x == y else 1 - h)

    # to networkx
    A = adjacency(g)
    G = nx.from_scipy_sparse_matrix(A)
    nx.set_node_attributes(G, label, {i: b for i, b in enumerate(bm)})

    # tunning homphily using degree 0 nodes
    nodes = [n for n in G.nodes() if G.degree(n) == 0]
    G = _add_edges(G, label, nodes, h)
    homophily = _get_homophily(G, label)
    diff = abs(round(homophily[1], 2) - h)

    # tunning homphily using all nodes
    tries = 0
    maxtries = 50
    while diff >= 0.05 and tries < maxtries:
        G = _add_edges(G, label, G.nodes(), h)
        homophily = _get_homophily(G, label)
        diff = abs(round(homophily[1], 2) - h)
        tries+=1
        print('B{} | H{} | tries{} | diff{}'.format(b,h,tries,diff))

    if diff >= 0.05:
        print('NOT good B:{} H{}, h{}'.format(b,h,homophily[1]))
        return
    else:
        homophily = _get_homophily(G, label)
        degrees = _get_degrees(G, label)
        name = '{}-B{}-H{}-h{}-k{}-{}'.format(prefix, b, h, round(homophily[1], 2), round(degrees[0], 1), round(degrees[1], 1))
        print(name)
        G.graph['attributes'] = [label]
        G.name = name
        info(G, label)
        save(G, name, path)
        plot(G, name, label, path)
        draw_adjacency_matrix(G, name, path)
        fn = os.path.join(path, name)
        graph_draw(g, vertex_fill_color=bm, edge_color="black", output_size=(500, 500), output=os.path.join(path, '{}-network2.png'.format(fn)))
        plot_degree(G, label, '{}-degree.png'.format(fn))
        plot_population(G, label, '{}-population.png'.format(fn))

def run_stochastic_block_models(label, prefix, output):
    path = os.path.join(output, prefix)
    create_path(path)

    N = 1000
    C = 2
    homophily = np.arange(0.1,1.1,0.1)       # homophily
    B = np.arange(0.1,0.6,0.1)   # class balance
    degree = 20

    runs = 5
    Parallel(n_jobs=runs)(delayed(sbm)(h,b,degree,label,N,C,prefix,path) for h in homophily for b in B)

def run_configuration_model(label, prefix, output):
    path = os.path.join(output, prefix)
    create_path(path)

    N = 1000
    C = 2
    K = [1,6,12,24]
    K1K2 = list(combinations_with_replacement(K,2))     # degrees
    B = np.arange(0.1, 1.0, 0.1)                       # class balance

    runs = 5
    Parallel(n_jobs=runs)(delayed(cnfmodel)(k1k2, b, label, N, C, prefix, path) for k1k2 in K1K2 for b in B)

def cnfmodel(k1k2, b, label, N, C, prefix, path):

    name = '{}-B{}-K{}-{}'.format(prefix, b, k1k2[0], k1k2[1])
    files = [f for f in os.listdir(path) if f.startswith(name)]

    if len(files) == 5:
        print('B{} | K{}-{} | already exists!'.format(b, k1k2[0], k1k2[1]))
        return
    else:
        for f in files:
            fn = os.path.join(path, f)
            print('to delete: {}'.format(fn))
            os.remove(fn)

    if C != 2:
        print('Support only for 2 groups.')
        sys.exit(0)

    size_1 = int(round(b*N))
    size_2 = N - size_1

    bm = {v: int(v >= size_1) for v in range(N)}
    print(Counter(bm.values()))

    print('*** Configuration Graph | B{} | K{} ***'.format(b,k1k2))

    while True:

        z = []
        z1 = np.random.poisson(k1k2[0], size_1)
        z2 = np.random.poisson(k1k2[1], size_2)
        z.extend(z1.tolist())
        z.extend(z2.tolist())

        if nx.is_valid_degree_sequence(z):
            #1. Configuration model graph
            G = nx.configuration_model(z)
            G = nx.Graph(G)
            G.remove_edges_from(G.selfloop_edges())

            nx.set_node_attributes(G, label, bm)
            nodes = [n for n in G.nodes() if G.degree(n)==0]

            for source in nodes:
                degrees = _get_degrees(G, label)

                if degrees[0] < k1k2[0] and degrees[1] < k1k2[1]:
                    while True:
                        target = np.random.choice(G.nodes(),1)[0]
                        if not G.has_edge(source,target):
                            G.add_edge(source,target)
                            break

                elif (degrees[0] < k1k2[0] and degrees[1] >= k1k2[1]) or (degrees[0] >= k1k2[0] and degrees[1] < k1k2[1]):
                    while True:
                        target = np.random.choice([n for n in G.nodes() if G.node[n][label]==G.node[source][label]],1)[0]
                        if not G.has_edge(source,target):
                            G.add_edge(source,target)
                            break
                else:
                    while True:
                        target = np.random.choice([n for n in G.nodes() if G.node[n][label]!=G.node[source][label]],1)[0]
                        if not G.has_edge(source,target):
                            G.add_edge(source,target)
                            break

            homophily = _get_homophily(G, label)
            degrees = _get_degrees(G, label)
            name = '{}-B{}-K{}-{}-h{}-k{}-{}'.format(prefix, b, k1k2[0], k1k2[1], round(homophily[1], 2), round(degrees[0], 1), round(degrees[1], 1))

            print(name)
            G.graph['attributes'] = [label]
            G.name = name
            info(G, label)
            save(G, name, path)
            plot(G, name, label, path)
            draw_adjacency_matrix(G, name, path)
            fn = os.path.join(path, name)
            plot_degree(G, label, '{}-degree.png'.format(fn))
            plot_population(G, label, '{}-population.png'.format(fn))
            print('')
            break

def _get_homophily(G, label):
    homophily = Counter([int(G.node[edge[0]][label] == G.node[edge[1]][label]) for edge in G.edges()])
    homophily = {k: v / float(sum(homophily.values())) for k, v in homophily.items()}

    if 0 not in homophily:
        homophily[0] = 0
    if 1 not in homophily:
        homophily[1] = 0

    return homophily

def _get_degrees(G, label):
    block_nodes = {n[0]: n[1][label] for n in G.nodes(data=True)}
    blocks = set(block_nodes.values())
    degrees = {}
    for block in blocks:
        ks = [G.degree(n) for n, b in block_nodes.items() if b == block]
        k = np.mean(ks)
        # print('Average degree {} {}: {}'.format(label, block, k))
        degrees[block] = k
    return degrees


#########################################
# CONF MODEL (DEGREE + BALANCE)
#########################################

def run_configuration_model_degree_balance(label, prefix, output):
    path = os.path.join(output, prefix)
    create_path(path)

    N = 1000
    C = 2
    B = np.arange(0.1, 1.1, 0.1)
    K = [(6,6)] #[(6,6), (6,24), (24,6)]
    print('Parameters (Degree+Balance):\n{}'.format('\n'.join(['B{}-K{}'.format(b, k1k2) for b in B for k1k2 in K])))
    runs = 1
    Parallel(n_jobs=runs)(delayed(cnfmodel_degreebalance)(b, k1k2, label, N, C, prefix, path) for b in B for k1k2 in K)

def cnfmodel_degreebalance(b, k1k2, label, N, C, prefix, path):

    if C != 2:
        print('Support only for 2 groups.')
        sys.exit(0)

    graphs = {'homophilic':None, 'heterophilic':None, 'neutral':None}
    size_1 = int(round(b*N))
    size_2 = N - size_1

    bm = {v: int(v >= size_1) for v in range(N)}
    print(Counter(bm.values()))

    mindegree = {6:5, 24:10}
    max_tries = 500
    tries = 1

    for k in graphs.keys():
        print('*** Configuration Graph | {} | B{} | K{} ***'.format(k.upper(), b,k1k2))
        tries = 1
        while tries <= max_tries:
            z = []
            z1 = np.random.poisson(k1k2[0], size_1)
            z2 = np.random.poisson(k1k2[1], size_2)
            z.extend(z1.tolist())
            z.extend(z2.tolist())

            if nx.is_valid_degree_sequence(z):
                #1. Configuration model graph
                G = nx.configuration_model(z)
                G = nx.Graph(G)
                G.remove_edges_from(G.selfloop_edges())

                nx.set_node_attributes(G, label, bm)
                homophily = _get_homophily(G, label)
                degrees = _get_degrees(G, label)

                # #1.1. validating degree
                # if abs(degrees[0]-k1k2[0]) <= mindegree[k1k2[0]] and abs(degrees[1]-k1k2[1]) <= mindegree[k1k2[1]]:
                #     # print('{}: B{} | K{}-{} | k{}-{} | GOOD | h{}'.format(tries, b,k1k2[0],k1k2[1],degrees[0],degrees[1],round(homophily[1], 2)))
                #     pass
                # else:
                #     # print('{}: B{} | K{}-{} | k{}-{} | BAD | h{}'.format(tries, b, k1k2[0], k1k2[1], degrees[0], degrees[1], round(homophily[1], 2)))
                #     continue

                # print('{} nodes with degree 0'.format(len([n for n in G.nodes() if G.degree(n) == 0])))
                # for b in set(bm.values()):
                #     print('{} nodes with degree 0 block {}'.format(len([n for n in G.nodes() if G.degree(n) == 0 and G.node[n][label] == b]),b))

                # 2. levaraging degree 0 nodes (validating homophily)
                if k == 'homophilic' and round(homophily[1], 2) < 0.7:
                    nodes = [n for n in G.nodes() if G.degree(n) == 0]
                    for node in nodes:
                        target = np.random.choice(G.nodes(),1)[0]
                        if G.node[target][label] == G.node[node][label] and not G.has_edge(node,target):
                            G.add_edge(node,target)

                elif k == 'heterophilic' and round(homophily[1], 2) > 0.3:
                    nodes = [n for n in G.nodes() if G.degree(n) == 0]
                    for node in nodes:
                        target = np.random.choice(G.nodes(), 1)[0]
                        if G.node[target][label] != G.node[node][label] and not G.has_edge(node, target):
                            G.add_edge(node, target)

                elif k == 'neutral':
                    nodes = [n for n in G.nodes() if G.degree(n) == 0]
                    for node in nodes:
                        target = np.random.choice(G.nodes(), 1)[0]

                        if round(homophily[1], 2) < 0.3:
                            if G.node[target][label] == G.node[node][label] and not G.has_edge(node, target):
                                G.add_edge(node, target)
                        elif round(homophily[1], 2) > 0.7:
                            if G.node[target][label] != G.node[node][label] and not G.has_edge(node, target):
                                G.add_edge(node, target)

                # shuffle
                homophily = _get_homophily(G, label)
                degrees = _get_degrees(G, label)

                if k == 'homophilic' and round(homophily[1], 2) < 0.7:
                    # more homophilic edges
                    edges = G.edges()
                    np.random.shuffle(edges)
                    edge = edges[0]

                    if G.node[edge[0]][label] != G.node[edge[1]][label]:
                        G.remove_edges_from([edge])

                        e = 0 if degrees[0] < k1k2[0] else 1 if degrees[1] < k1k2[1] else np.random.choice([0,1],1)[0]
                        target = np.random.choice([n for n in G.nodes() if G.node[n][label] == G.node[edge[e]][label] and n != edge[e]],1)[0]
                        G.add_edge(edge[e],target)

                elif k == 'heterophilic' and round(homophily[1], 2) > 0.3:
                    # more hetero edges
                    edges = G.edges()
                    np.random.shuffle(edges)
                    edge = edges[0]

                    if G.node[edge[0]][label] == G.node[edge[1]][label]:
                        G.remove_edges_from([edge])

                        e = 0 if degrees[0] < k1k2[0] else 1 if degrees[1] < k1k2[1] else np.random.choice([0,1],1)[0]
                        target = np.random.choice([n for n in G.nodes() if G.node[n][label] != G.node[edge[e]][label] and n != edge[e]],1)[0]
                        G.add_edge(edge[e],target)

                elif k == 'neutral':
                    edges = G.edges()
                    np.random.shuffle(edges)
                    edge = edges[0]

                    if round(homophily[1], 2) < 0.3:
                        if G.node[edge[0]][label] != G.node[edge[1]][label]:
                            G.remove_edges_from([edge])

                            e = 0 if degrees[0] < k1k2[0] else 1 if degrees[1] < k1k2[1] else np.random.choice([0, 1], 1)[0]
                            target = np.random.choice([n for n in G.nodes() if G.node[n][label] == G.node[edge[e]][label] and n != edge[e]], 1)[0]
                            G.add_edge(edge[e], target)

                    elif round(homophily[1], 2) > 0.7:
                        if G.node[edge[0]][label] != G.node[edge[1]][label]:
                            G.remove_edges_from([edge])

                            e = 0 if degrees[0] < k1k2[0] else 1 if degrees[1] < k1k2[1] else np.random.choice([0, 1], 1)[0]
                            target = np.random.choice([n for n in G.nodes() if G.node[n][label] == G.node[edge[e]][label] and n != edge[e]], 1)[0]
                            G.add_edge(edge[e], target)

                # end

                if graphs[k] is None:
                    graphs[k] = nx.Graph(G)
                else:
                    previous_homophily = _get_homophily(graphs[k], label)
                    if k=='homophilic' and homophily[1] > previous_homophily[1]:
                        graphs[k] = nx.Graph(G)
                    elif k=='heterophilic' and homophily[1] < previous_homophily[1]:
                        graphs[k] = nx.Graph(G)
                    elif k=='neutral' and homophily[1] < previous_homophily[1]:
                        graphs[k] = nx.Graph(G)

                # # homophily
                # if round(homophily[1], 2) >= 0.7:
                #     if graphs['homophilic'] is None:
                #         graphs['homophilic'] = nx.Graph(G)
                #     else:
                #         previous_homophily = _get_homophily(graphs['homophilic'], label)
                #         if homophily[1] > previous_homophily[1]:
                #             graphs['homophilic'] = nx.Graph(G)
                #
                # elif round(homophily[1], 2) <= 0.3:
                #     if graphs['heterophilic'] is None:
                #         graphs['heterophilic'] = nx.Graph(G)
                #     else:
                #         previous_homophily = _get_homophily(graphs['heterophilic'], label)
                #         if homophily[1] < previous_homophily[1]:
                #             graphs['heterophilic'] = nx.Graph(G)
                # else:
                #     if graphs['neutral'] is None:
                #         graphs['neutral'] = nx.Graph(G)
                #     else:
                #         previous_homophily = _get_homophily(graphs['neutral'], label)
                #         if homophily[1] < previous_homophily[1] and previous_homophily[1] >= 0.5:
                #             graphs['neutral'] = nx.Graph(G)

                # if graphs['homophilic'] is not None and graphs['heterophilic'] is not None and graphs['neutral'] is not None:
                #     break

                tries += 1

    print('*** Final Configuration Graph | B{} | K{} ***'.format(b, k1k2))

    for k,graph in graphs.items():
        if graph is None:
            print('There is no {} graph for B{} K{}'.format(k,b,k1k2))
            continue
        homophily = _get_homophily(graph, label)
        degrees = _get_degrees(graph, label)
        name = '{}-B{}-K{}-{}-H{}-h{}-k{}-{}'.format(prefix, b, k1k2[0], k1k2[1], k, round(homophily[1], 2), round(degrees[0], 1), round(degrees[1], 1))
        print('{}'.format(name))

        if abs(k1k2[0]-degrees[0]) > mindegree[k1k2[0]] \
                or abs(k1k2[1]-degrees[1]) > mindegree[k1k2[1]] \
                or ( k == 'homophilic' and homophily[1] < 0.7) \
                or ( k == 'heterophilic' and homophily[1] > 0.3) \
                or ( k == 'neutral' and (homophily[1] >= 0.7 or homophily[1] <= 0.3)):
            print('not valid.')
            pass
        else:
            graph.graph['attributes'] = [label]
            graph.name = name
            info(graph, label)
            save(graph, name, path)
            plot(graph, name, label, path)
            draw_adjacency_matrix(graph, name, path)
            print('')
        raw_input('...')

def check_configuration_model(label,prefix,output):
    path = os.path.join(output, prefix)

    N = 1000
    C = 2
    B = np.arange(0.1, 0.6, 0.1)
    H = np.arange(0.1, 1.1, 0.1)
    K = np.arange(6, 60, 6)
    degrees = list(product(K, repeat=C))
    suma = 0

    for counts in range(1,11):
        print('===== COUNTS:{} ====='.format(counts))
        for k in degrees:
            for b in B:

                folders = [x for x in os.listdir(path)
                           if x.lower().startswith('{}-'.format(prefix.lower()))
                           and x.endswith('.gpickle')
                           and '-B{}-'.format(b) in x
                           and '-K{}-{}-'.format(k[0],k[1]) in x]

                if len(folders) == counts:
                    print('k-{}-{} | b-{} | {} files'.format(k[0],k[1],b,len(folders)))
                    suma += len(folders)

    print('{} total files'.format(suma))
    return


############################################
# FARIBA (BA HOMOPHILY, MINORITY, MAJORITY)
############################################

def create_BAHM_graph(prefix, output, N=2000, m=4, B=0.5):

    ### PATH, OUTPUT
    path = os.path.join(output, prefix)
    create_path(path)
    print('Path created: {}'.format(path))
    ### INPUT, PARAMS
    H = [np.round(i,1) for i in np.arange(0.0, 1.1, 0.1)]

    print('Generating networks for N:{} and m:{} and B:{}'.format(N,m,B))

    try:
        x = int(prefix.split("x")[-1])
    except:
        x = 1
    print('{} generations.'.format(x))

    ### GENERATION
    _ = Parallel(n_jobs=10)(delayed(_create_BAHM_graph)(N,m,h,B,i,x,path) for i in np.arange(1,x+1,1) for h in H)

def _create_BAHM_graph(N,m,h,B,i,x,path):
    print('=== H:{} | B:{} | {} out of {} ==='.format(h, B, i, x))
    np.random.seed(None)
    graph = homophilic_barabasi_albert_graph(N=N, m=m, minority_fraction=B, similitude=h)

    k = round(2 * graph.number_of_edges() / float(graph.number_of_nodes()),1) # undirected
    name = '{}-N{}-B{}-H{}-PK{}-i{}'.format(prefix, N, B, h, k, i)

    stats_BAHM2(graph)
    save_BAHM2(graph,name,'color',path)
    print('')

def save_BAHM2(G,name,label,path):

    homophily = _get_homophily(G, label)
    degrees = _get_degrees(G, label)
    name = '{}-h{}-k{}-{}'.format(name, round(homophily[1], 2), round(degrees['red'], 1), round(degrees['blue'], 1))

    # networkx
    G.graph['attributes'] = [label]
    G.name = name

    info(G, label)
    save(G, name, path)
    plot(G, name, label, path)

    values = nx.get_node_attributes(G, label)
    nodes = [node for node, value in sorted(values.items(), key=lambda item: (item[1], item[0]))]
    draw_adjacency_matrix(G, name, path, nodes)

    fn = os.path.join(path,name)
    plot_degree(G, label, '{}-degree.png'.format(fn))
    plot_population(G, label, '{}-population.png'.format(fn))
    plot_degree_degree(G, label, '{}-degreedegree-matrixnorm.png'.format(fn), True)
    plot_degree_degree(G, label, '{}-degreedegree.png'.format(fn), False)

def stats_BAHM2(graph):
    ### balance
    attribute = nx.get_node_attributes(graph, 'color')
    tmp = Counter([c for n, c in attribute.items()])
    print('Class Balance: {}'.format(' | '.join(['{}:{}'.format(k, v / float(graph.number_of_nodes())) for k, v in tmp.items()])))

    ### homophily
    homophily = sum([int(graph.node[edge[0]]['color'] == graph.node[edge[1]]['color']) for edge in graph.edges()])
    print('Homophily: {}'.format(homophily / float(graph.number_of_edges())))

##################################################
# For testing
##################################################

def _powerlaw(output):
    N = 1000
    plt.close('all')
    alphas = [(1.5,3.5),(3.5,1.5),(2.0,2.0)]
    classbalance = [0.2,0.5]

    for b in classbalance:
        f, axarr = plt.subplots(2, len(alphas)*2, figsize=(30, 10))

        for c,a in enumerate(alphas):
            size_1 = int(round(b*N))
            size_2 = N - size_1

            theoretical_distribution_1 = powerlaw.Power_Law(xmin=1., parameters=[a[0]], discrete=True)
            simulated_data_1 = theoretical_distribution_1.generate_random(size_1).astype(int)
            fit_1 = powerlaw.Fit(simulated_data_1)
            print('Fitted 1 | xmin:{} alpha:{}'.format(fit_1.power_law.xmin, fit_1.power_law.alpha))
            powerlaw.plot_pdf(simulated_data_1, axarr[0, c*2], linewidth=3)
            fit_1.power_law.plot_pdf(simulated_data_1, axarr[0, c*2], linestyle='--', color='r')
            simulated_data_1 = simulated_data_1.tolist()
            x1 = sorted(set(simulated_data_1))
            y1 = [simulated_data_1.count(d) for d in x1]
            axarr[1, c*2].loglog(x1, y1)
            axarr[1, c*2].set_ylim(ymin=0.02)
            axarr[0, c*2].set_title('alpha:{} | b:{} | xmin:{} | alpha:{}'.format(a[0],b,fit_1.power_law.xmin, round(fit_1.power_law.alpha,2)))

            theoretical_distribution_2 = powerlaw.Power_Law(xmin=1., parameters=[a[1]], discrete=True)
            simulated_data_2 = theoretical_distribution_2.generate_random(size_2).astype(int)
            fit_2 = powerlaw.Fit(simulated_data_2)
            print('Fitted 2 | xmin:{} alpha:{}'.format(fit_2.power_law.xmin, fit_2.power_law.alpha))
            powerlaw.plot_pdf(simulated_data_2, axarr[0, (c * 2)+1], linewidth=3)
            fit_2.power_law.plot_pdf(simulated_data_2, axarr[0, (c * 2)+1], linestyle='--', color='r')
            simulated_data_2 = simulated_data_2.tolist()
            x2 = sorted(set(simulated_data_2))
            y2 = [simulated_data_2.count(d) for d in x2]
            axarr[1, (c * 2)+1].loglog(x2, y2)
            axarr[1, (c * 2)+1].set_ylim(ymin=0.02)
            axarr[0, (c * 2)+1].set_title('alpha:{} | b:{} | xmin:{} | alpha:{}'.format(a[1], 1.-b, fit_2.power_law.xmin, round(fit_2.power_law.alpha, 2)))

        plt.suptitle('Powerlaw dist.')
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)

        p = int(round((b*100)))
        fn = os.path.join(output,'powerlaw_{}-{}.png'.format(p,100-p))
        plt.savefig(fn)
        plt.close()

def sbm_poisson():
    g = collection.data["polblogs"]
    g = GraphView(g, vfilt=label_largest_component(g))
    g = Graph(g, prune=True)

    state = minimize_blockmodel_dl(g)

    print('state.b\n{}'.format(state.b))
    print('state.b.a\n{}'.format(state.b.a))
    print('state.b.a type\n{}'.format(type(state.b.a)))
    print('state.b.a shape\n{}'.format(state.b.a.shape))

    print('state.get_ers()\n{}'.format(state.get_ers()))
    print('state.get_bg()\n{}'.format(state.get_bg()))
    print('adjacency\n{}'.format(adjacency(state.get_bg(), state.get_ers())))
    print('adjacency type\n{}'.format(type(adjacency(state.get_bg(), state.get_ers()))))

    print('in-degree\n{}'.format(g.degree_property_map("in").a))
    print('out-degree\n{}'.format(g.degree_property_map("out").a))

    u = generate_sbm(state.b.a,
                     adjacency(state.get_bg(),state.get_ers()),
                     g.degree_property_map("out").a,
                     g.degree_property_map("in").a,
                     directed = True)

    graph_draw(g, g.vp.pos, output="polblogs-sbm.png")
    graph_draw(u, g.vp.pos, output="polblogs-sbm-generated.png")

def sbm_poisson_powerlaw(label, prefix, path):
    path = os.path.join(output, prefix)
    create_path(path)

    B = 2
    N = 100
    plt.close('all')
    alphas = [(1.5, 3.5), (3.5, 1.5), (2.0, 2.0)]
    classbalance = [0.2, 0.5]
    homophily = [0.2, 0.5, 0.8]

    for b in classbalance:
        for h in homophily:
            for c, a in enumerate(alphas):
                size_1 = int(round(b * N))
                size_2 = N - size_1

                node_blocks = np.array([int(v < size_1) for v in range(N)])

                theoretical_distribution_1 = powerlaw.Power_Law(xmin=1., parameters=[a[0]], discrete=True)
                simulated_data_1 = theoretical_distribution_1.generate_random(size_1).astype(int)
                fit_1 = powerlaw.Fit(simulated_data_1)
                print('Fitted 1 | xmin:{} alpha:{}'.format(fit_1.power_law.xmin, fit_1.power_law.alpha))

                theoretical_distribution_2 = powerlaw.Power_Law(xmin=1., parameters=[a[1]], discrete=True)
                simulated_data_2 = theoretical_distribution_2.generate_random(size_2).astype(int)
                fit_2 = powerlaw.Fit(simulated_data_2)
                print('Fitted 2 | xmin:{} alpha:{}'.format(fit_2.power_law.xmin, fit_2.power_law.alpha))

                degrees = simulated_data_1.tolist()
                degrees.extend(simulated_data_2.tolist())

                sumdegrees = sum(degrees)
                block_adj = np.zeros((B,B)) + sumdegrees
                block_adj[0, 0] *= h
                block_adj[1, 1] *= h
                block_adj[0, 1] *= (1.-h)
                block_adj[1, 0] *= (1.-h)

                key = '{}_H{}_B{}_A{}-{}'.format(prefix, int(h * 100), int(b * 100), a[0], a[1] )
                print('')
                print('=================== {} ==================='.format(key))
                print('block1: shape:{} | size:{}'.format(simulated_data_1.shape,simulated_data_1.size))
                print('block2: shape:{} | size:{}'.format(simulated_data_2.shape, simulated_data_2.size))
                print('sumdegrees: {}'.format(sumdegrees))
                print('block_adj: \n{}'.format(block_adj))
                print('node_blocks: \n{}'.format(node_blocks))
                print('degrees: \n{}'.format(degrees))

                g = generate_sbm(node_blocks,
                                 block_adj,
                                 degrees,
                                 degrees,
                                 directed=False)


                # to networkx
                A = adjacency(g)
                G = nx.from_scipy_sparse_matrix(A)
                nx.set_node_attributes(G, label, {i: b for i, b in enumerate(node_blocks.tolist())})
                G.graph['attributes'] = [label]

                # info
                G.name = key
                save(G, key, path)
                plot(G, key, label, path)
                info(G, label)
                draw_adjacency_matrix(G, key, path)
                graph_draw(g, output=os.path.join(path, '{}.png'.format(key)))
                graph_draw(g, vertex_fill_color=node_blocks.tolist(), edge_color="black", output_size=(500, 500), output=os.path.join(path, '{}-network2.png'.format(key)))

    # N = 1000
    #
    # B = 2
    # class_balance = [0.2, 0.5]
    # degrees = [10.0, 100.]
    # homophily = [0.8,0.5]
    # alpha = 0.2
    # degrees = list(product(degrees,repeat=2))
    #
    # print('degrees: {}'.format(degrees))
    #
    # for block in class_balance:
    #     population = int(round(N * block))
    #     node_blocks = np.array([int(v < population) for v in range(N)])
    #
    #     for h in homophily:
    #         for d in degrees:
    #
    #             degree = [d[b] for b in node_blocks]
    #
    #             probs = np.zeros((B,B))
    #             probs[0, :] = h * sum([degree[v] for v,b in enumerate(node_blocks) if b == 0]) / sum([1. for b in node_blocks if b == 0])
    #             probs[1, :] = (1.-h) * sum([degree[v] for v,b in enumerate(node_blocks) if b == 1]) / sum([1. for b in node_blocks if b == 1])
    #
    #             print('Block:{} | Homophily:{} | Degree:{}\nProbs:{}\nNode Blocks:{}\nDegree:{}'.format(block, h, d, probs, node_blocks, degree))
    #
    #             node_blocks = node_blocks.reshape(-1,1)
    #             print(node_blocks)
    #             print(node_blocks.shape)
    #
    #             g, bm = generate_sbm(N,
    #                      b=node_blocks,
    #                      probs=probs,
    #                      in_degs=degree,
    #                      out_degs=degree,
    #                      directed=False)
    #
    #             # to networkx
    #             A = adjacency(g)
    #             G = nx.from_scipy_sparse_matrix(A)
    #             nx.set_node_attributes(G, label, {i: b for i, b in enumerate(bm)})
    #             G.graph['attributes'] = [label]
    #
    #             # info
    #             type = '{}_H{}_B{}_S{}'.format(prefix, int(h * 100), int(block * 100), '{}'.format('-'.join([str(s).replace('.', '') for s in d])))
    #             G.name = type
    #             save(G, type, path)
    #             plot(G, type, label, path)
    #             info(G, label)
    #             draw_adjacency_matrix(G, type, path)
    #             graph_draw(g, vertex_fill_color=bm, edge_color="black", output_size=(500, 500), output=os.path.join(path, '{}-network2.png'.format(type)))



    return

##################################################
# Graph
##################################################
def toy():
    G = nx.Graph()

    # nodes
    G.add_node(1, {'gender': 'f', 'age': 19, 'status': 'a'})
    G.add_node(2, {'gender': 'f', 'age': 19, 'status': 'b'})
    G.add_node(3, {'gender': 'f', 'age': 18, 'status': 'b'})
    G.add_node(4, {'gender': 'f', 'age': 18, 'status': 'a'})
    G.add_node(5, {'gender': 'f', 'age': 19, 'status': 'b'})
    G.add_node(6, {'gender': 'f', 'age': 19, 'status': 'b'})
    G.add_node(7, {'gender': 'f', 'age': 18, 'status': 'b'})
    G.add_node(8, {'gender': 'm', 'age': 60, 'status': 'a'})
    G.add_node(9, {'gender': 'm', 'age': 60, 'status': 'a'})
    G.add_node(10, {'gender': 'm', 'age': 50, 'status': 'a'})
    G.add_node(11, {'gender': 'm', 'age': 50, 'status': 'a'})
    G.add_node(12, {'gender': 'f', 'age': 19, 'status': 'b'})
    G.add_node(13, {'gender': 'f', 'age': 18, 'status': 'b'})
    G.add_node(14, {'gender': 'f', 'age': 18, 'status': 'a'})
    G.add_node(15, {'gender': 'f', 'age': 19, 'status': 'b'})
    G.add_node(16, {'gender': 'f', 'age': 19, 'status': 'b'})
    G.add_node(17, {'gender': 'f', 'age': 18, 'status': 'b'})
    G.add_node(18, {'gender': 'm', 'age': 60, 'status': 'a'})
    G.add_node(19, {'gender': 'm', 'age': 60, 'status': 'a'})
    G.add_node(20, {'gender': 'm', 'age': 50, 'status': 'a'})

    # edges
    G.add_edge(1, 2)
    G.add_edge(1, 3)
    G.add_edge(1, 4)
    G.add_edge(3, 4)
    G.add_edge(2, 5)
    G.add_edge(2, 6)
    G.add_edge(3, 7)
    G.add_edge(4, 8)
    G.add_edge(4, 9)
    G.add_edge(8, 9)
    G.add_edge(9, 10)
    G.add_edge(9, 11)
    G.add_edge(10, 11)
    G.add_edge(1, 12)
    G.add_edge(1, 13)
    G.add_edge(1, 14)
    G.add_edge(13, 14)
    G.add_edge(12, 15)
    G.add_edge(12, 16)
    G.add_edge(13, 17)
    G.add_edge(14, 8)
    G.add_edge(14, 9)
    G.add_edge(18, 19)
    G.add_edge(19, 20)
    G.add_edge(19, 11)
    G.add_edge(10, 20)

    G.graph['attributes'] = ['gender', 'age', 'status']

def colored(type):
    BLOCKS = 2
    nnodes = 100
    walks = 2 * nnodes
    LOW = 0.01
    HIGH = 0.99
    RANDOM = 0.5
    if type == HOMOPHILIC:
        COLORPROB = {'red': {'red': HIGH, 'blue': LOW}, 'blue': {'red': LOW, 'blue': HIGH}}
    elif type == HETEROPHILIC:
        COLORPROB = {'red': {'red': LOW, 'blue': HIGH}, 'blue': {'red': HIGH, 'blue': LOW}}
    elif type == NEUTRAL:
        COLORPROB = {'red': {'red': RANDOM, 'blue': RANDOM}, 'blue': {'red': RANDOM, 'blue': RANDOM}}
    COLORVOCAB = {0: 'red', 1: 'blue'}
    colordistribution = {}

    G = nx.Graph()
    G.name = type
    nodes = {n: int(n * BLOCKS / nnodes) for n in range(nnodes)}

    for source, block in nodes.items():
        if source not in G:
            G.add_node(source, color=COLORVOCAB[block])

        for i in range(walks):
            target = None
            while (target == source or target is None):
                target = randint(0, nnodes - 1)

            if target not in G:
                G.add_node(target, color=COLORVOCAB[nodes[target]])

            prob = COLORPROB[COLORVOCAB[block]][COLORVOCAB[nodes[target]]]
            draw = np.random.binomial(n=1, p=prob, size=1)

            if draw[0] == 1:
                G.add_edge(source, target, weight=1.)
                if COLORVOCAB[block] not in colordistribution:
                    colordistribution[COLORVOCAB[block]] = {}
                if COLORVOCAB[nodes[target]] not in colordistribution[COLORVOCAB[block]]:
                    colordistribution[COLORVOCAB[block]][COLORVOCAB[nodes[target]]] = 0
                colordistribution[COLORVOCAB[block]][COLORVOCAB[nodes[target]]] += 1

    G.graph['attributes'] = ['color']
    return G

##################################################
# Functions
##################################################
def plot(G, fn, label, output):

    labels = list(set([G.node[n][label] for n in G.nodes()]))
    colors = iter(cm.rainbow(np.linspace(0, 1, len(labels) + 1)))
    pos = nx.spring_layout(G)

    _ = nx.draw_networkx_edges(G, pos, alpha=0.2)
    for l in labels:
        c = next(colors)
        nodes = [n for n in G.nodes() if G.node[n][label] == l]
        _ = nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=[c] * len(nodes), with_labels=False, cmap=plt.cm.jet, label=l)
    plt.legend()
    plt.title(label.upper())
    plt.axis('off')
    plt.savefig(os.path.join(output,'{}-network.png'.format(fn)))
    plt.close()

def create_path(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except Exception as ex:
            print(ex)

def info(G, label):
    print(nx.info(G))

    # average degree per block
    block_nodes = {n[0]:n[1][label] for n in G.nodes(data=True)}
    blocks = set(block_nodes.values())
    degrees = {}
    for block in blocks:
        ks = [G.degree(n) for n,b in block_nodes.items() if b == block]
        k = np.mean(ks)
        print('Average degree {} {}: {}'.format(label,block,k))
        degrees[block] = k

    da = nx.degree_assortativity_coefficient(G)
    aa = nx.attribute_assortativity_coefficient(G, label)
    print('degree assortativity: {}'.format(da))
    print('attribute assortativity: {}'.format(aa))

    homophily = Counter([G.node[edge[0]][label] == G.node[edge[1]][label] for edge in G.edges()])
    homophily = {k: v / float(sum(homophily.values())) for k, v in homophily.items()}
    print('(edge) homophily: {}'.format(homophily))

    return homophily, degrees

def save(G, fn , path):
    fn = os.path.join(path, '{}.gpickle'.format(fn))
    nx.write_gpickle(G, fn)
    print('{} saved!'.format(fn))

def draw_adjacency_matrix(G, fn, path, node_order=None, partitions=[], colors=[]):
    """
    - G is a netorkx graph
    - node_order (optional) is a list of nodes, where each node in G
          appears exactly once
    - partitions is a list of node lists, where each node in G appears
          in exactly one node list
    - colors is a list of strings indicating what color each
          partition should be
    If partitions is specified, the same number of colors needs to be
    specified.
    """
    adjacency_matrix = nx.to_numpy_matrix(G, dtype=np.bool, nodelist=node_order)

    # Plot adjacency matrix in toned-down black and white
    fig = pyplot.figure(figsize=(5, 5))  # in inches
    pyplot.imshow(adjacency_matrix,
                  cmap="Greys",
                  interpolation="none")

    # The rest is just if you have sorted nodes by a partition and want to
    # highlight the module boundaries
    assert len(partitions) == len(colors)
    ax = pyplot.gca()
    for partition, color in zip(partitions, colors):
        current_idx = 0
        for module in partition:
            ax.add_patch(patches.Rectangle((current_idx, current_idx),
                                           len(module),  # Width
                                           len(module),  # Height
                                           facecolor="none",
                                           edgecolor=color,
                                           linewidth="1"))
            current_idx += len(module)

    pyplot.savefig(os.path.join(path,'{}-adjacency.png'.format(fn)))
    pyplot.close()

def read_sbm(fn):
    path = os.path.dirname(os.path.abspath(fn))
    name = os.path.basename(fn)

    tmp = [f for f in os.listdir(path) if f.startswith(name) and f.endswith('.p')]
    if len(tmp) == 1:
        fn = os.path.join(path,tmp[0])

    if not fn.endswith('.p'):
        fn = '{}.p'.format(fn)

    if os.path.exists(fn):
        with open(fn,'rb') as f:
            obj = pickle.load(f)
        print('{} loaded!'.format(fn))
        return obj[0], obj[1]
    print('{} ERROR loading.'.format(fn))
    return None, None

def save_sbm(g,bm,fn):
    if not fn.endswith('.p'):
        fn = '{}.p'.format(fn)

    try:
        with open(fn, 'wb') as f:
            pickle.dump([g,bm],f)
        print('{} saved!'.format(fn))
    except:
        return False
        print('{} NOT saved.'.format(fn))
    return True

def plot_degree_degree(G,label,fn,norm):
    edges = [(G.degree(edge[0]), G.degree(edge[1])) for edge in G.edges()]
    if not G.is_directed():
        edges.extend([(G.degree(edge[1]), G.degree(edge[0])) for edge in G.edges()])
    sources, targets = zip(*edges)

    df = pd.DataFrame({'source': sources, 'target': targets})
    df = df.groupby(['source', 'target']).size()
    df = df.unstack().fillna(0)
    df.sort_index()
    xticks = df.index
    yticks = df.columns

    if norm:
        if 'norm' not in fn:
            fn = fn.replace('.png','-norm.png').replace('.pdf','-norm.pdf')
        df /= df.values.sum()
    else:
        df = df.values

    fig, ax = plt.subplots()
    plt.pcolor(df, cmap='Blues')
    plt.colorbar()
    plt.title(label.upper())

    minorLocator = AutoMinorLocator()
    ax.xaxis.set_minor_locator(minorLocator)
    plt.tick_params(which='both', width=1)
    plt.tick_params(which='major', length=5)
    plt.tick_params(which='minor', length=2, color='g')

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x,pos: xticks[int(x)] if x < len(xticks) else ''))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: xticks[int(x)] if x < len(yticks) else ''))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    ax.set_xlim(xticks.min(), xticks.max())
    ax.set_ylim(yticks.min(), yticks.max())

    ax.axis('tight')
    plt.tight_layout()
    plt.savefig(fn)
    plt.close()


##################################################
# Main
##################################################
if __name__ == '__main__':

    output = sys.argv[1]
    N = int(sys.argv[2])
    m = int(sys.argv[3])
    B = float(sys.argv[4])
    x = int(sys.argv[5])

    prefix = '{}{}N{}{}'.format(BAHM,m,N,'' if x is None else 'x{}'.format(x))
    if os.path.exists(os.path.join(output,prefix)):
        print('{} already exists.'.format(prefix))
        #sys.exit(0)

    ### FARIBA
    create_BAHM_graph(prefix, output, N=N, m=m, B=B) #m: 4 sparse, 20 dense


    ### SBM
    # create_sbm_graph(SBM, 'block', output)

    # ### CONFIHURATION RM
    # if var == 'degree':
    #     run_configuration_model('block', CONFIGURATION_MODEL_POISSON, output)
    # elif var == 'homophily':
    #     run_stochastic_block_models('block', CONFIGURATION_MODEL_POISSON, output)

    ### STOCHASTIC BLOCK MODEL
    # create_sbm_graph(SBM,'block',output)
    # run_pseudo_sbm('block',PSEUDO_SBM,output)
    # run_stochastic_2block_models(BLOCKMODEL_POISSON,output)
    # _powerlaw(output)
    # sbm_poisson('block', BLOCKMODEL_POISSON, output)
    # sbm_poisson_powerlaw('block', BLOCKMODEL_PPL, output)

    ### COLORED
    # run_colored_graphs()

    # postfix = sys.argv[2]

    # if id not in [1,2,3,4,5]:
    #     print('id must be between 1 and 5.')
    #     sys.exit(0)

    # prefix = '{}_{}'.format(BAHM2,id)
    #var = str(sys.argv[2]).lower() # degree, homophily

    # if var not in ['degree','homophily']:
    #     sys.exit(0)


