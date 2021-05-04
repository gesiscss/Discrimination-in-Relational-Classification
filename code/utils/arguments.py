import argparse

def init_batch_create_network():
    parser = argparse.ArgumentParser()

    parser.add_argument('-kind', action='store',
                        dest='kind',
                        choices=["BAH"],
                        required=True,
                        help='Type of network.')

    parser.add_argument('-N', action='store',
                        dest='N',
                        type=int,
                        default=None,
                        help='Number of nodes.')

    parser.add_argument('-m', action='store',
                        dest='m',
                        type=int,
                        default=None,
                        help='Minimun degree')

    parser.add_argument('-B', action='store',
                        dest='B',
                        type=float,
                        default=None,
                        help='Minority fraction (class balance)')

    parser.add_argument('-density', action='store',
                        dest='density',
                        type=float,
                        default=None,
                        help='Edge density')

    parser.add_argument('-H', action='store',
                        dest='H',
                        type=float,
                        default=None,
                        help='Homophily symmetric')

    parser.add_argument('-Hmm', action='store',
                        dest='Hmm',
                        type=float,
                        default=None,
                        help='Homophily among minorities (asymmetric)')

    parser.add_argument('-HMM', action='store',
                        dest='HMM',
                        type=float,
                        default=None,
                        help='Homophily among majorities (asymmetric)')

    parser.add_argument('-fit', action='store',
                        dest='fit',
                        default=None,
                        help='Empirical network file (.gpickle)')
    
    parser.add_argument('-evalues', action='store',
                        dest='evalues',
                        default=None,
                        help='File containing homophily values based on E** (.csv)')
    
    parser.add_argument('-Hsym', action='store_true',
                        dest='Hsym',
                        default=True,
                        required=True,
                        help='Use symmetric homophily or not.')
    
    parser.add_argument('-i', action='store',
                        dest='i',
                        type=int,
                        default=1,
                        help='Network id (i out of x)')

    parser.add_argument('-x', action='store',
                        dest='x',
                        type=int,
                        default=5,
                        help='Max number of networks (i out of x)')

    parser.add_argument('-ignoreInt', action='store',
                        dest='ignoreInt',
                        type=int,
                        help='Class value to ignore (as int).')

    parser.add_argument('-root', action='store',
                        dest='root',
                        required=True,
                        help='Directory to store all networks.')

    parser.add_argument('--version', action='version', version='%(prog)s 1.0')

    results = parser.parse_args()

    print("===================================================")
    print("= ARGUMENTS PASSED:                               =")
    print("===================================================")
    print('kind ....... = ', results.kind)
    print('fit ........ = ', results.fit)
    print('N .......... = ', results.N)
    print('m .......... = ', results.m)
    print('B .......... = ', results.B)
    print('density .... = ', results.density)
    print('H .......... = ', results.H)
    print('Hmm ........ = ', results.Hmm)
    print('HMM ........ = ', results.HMM)
    print('i .......... = ', results.i)
    print('x .......... = ', results.x)
    print('ignoreInt .. = ', results.ignoreInt)
    print('root ....... = ', results.root)
    print("===================================================")

    return results

def init_batch_summary():
    parser = argparse.ArgumentParser()

    parser.add_argument('-kind', action='store',
                        dest='kind',
                        required=True,
                        choices=["all","BAH","Caltech","Escorts","Swarthmore","USF","Wikipedia"],
                        help='Kind of networkx graph.')

    parser.add_argument('-LC', action='store',
                        dest='LC',
                        choices=["prior"],
                        required=True,
                        help='Local classifier.')

    parser.add_argument('-RC', action='store',
                        dest='RC',
                        choices=["nBC"],
                        required=True,
                        help='Relational classifier.')

    parser.add_argument('-CI', action='store',
                        dest='CI',
                        choices=["relaxation"],
                        required=True,
                        help='Collective inference algorithm,')

    parser.add_argument('-sampling', action='store',
                        dest='sampling',
                        required=True,
                        choices=["all","nodes", "nedges", "degree", "neighbors", "partial_crawls"],
                        help='Sampling method (nodes, nedges, degree, neighbors, partial_crawls, all).',
                        )

    parser.add_argument('-njobs', action='store',
                        dest='njobs',
                        type=int,
                        default=1,
                        help='Number of parallel jobs.')

    parser.add_argument('-output', action='store',
                        dest='output',
                        required=True,
                        help='Directory to store all results.')

    parser.add_argument('-overwrite', action='store_true',
                        dest='overwrite',
                        default=False,
                        help='Overwrite or not file.')

    parser.add_argument('--version', action='version', version='%(prog)s 1.0')

    results = parser.parse_args()

    print("===================================================")
    print("= ARGUMENTS PASSED:                               =")
    print("===================================================")
    print('kind ....... = ', results.kind)
    print('LC ......... = ', results.LC)
    print('RC ......... = ', results.RC)
    print('CI ......... = ', results.CI)
    print('sampling ... = ', results.sampling)
    print('njobs ...... = ', results.njobs)
    print('output ..... = ', results.output)
    print('overwrite .. = ', results.overwrite)
    print("===================================================")

    return results

def init_batch_mixed_effects():
    parser = argparse.ArgumentParser()

    parser.add_argument('-kind', action='store',
                        dest='kind',
                        required=True,
                        choices=["BAH"],
                        help='Network source or kind (only synthetic).',
                        )

    parser.add_argument('-sampling', action='store',
                        dest='sampling',
                        required=True,
                        choices=["nodes", "nedges", "degree", "neighbors", "partial_crawls"],
                        help='Sampling method (nodes, nedges, degree, neighbors, partial_crawls).',
                        )

    parser.add_argument('-groups', action='store',
                        dest='groups',
                        required=True,
                        choices=["Hp"],
                        help='Groups for random effects.',
                        )

    parser.add_argument('-output', action='store',
                        dest='output',
                        required=True,
                        help='Directory to store all results.')

    parser.add_argument('-njobs', action='store',
                        dest='njobs',
                        type=int,
                        default=1,
                        help='Number of parallel jobs.')

    parser.add_argument('-verbose', action='store_true',
                        dest='verbose',
                        default=False,
                        help='To print out or not')

    parser.add_argument('--version', action='version', version='%(prog)s 1.0')

    results = parser.parse_args()

    print("===================================================")
    print("= ARGUMENTS PASSED:                               =")
    print("===================================================")
    print('datafn ..... = ', results.kind)
    print('sampling ... = ', results.sampling)
    print('groups ..... = ', results.groups)
    print('njobs ...... = ', results.njobs)
    print('verbose .... = ', results.verbose)
    print('output ..... = ', results.output)
    print("===================================================")

    return results

def init_batch_collective_classification():
    parser = argparse.ArgumentParser()

    parser.add_argument('-datafn', action='store',
                        dest='datafn',
                        required = True,
                        help='Path to gpickle networkx graph.')

    parser.add_argument('-LC', action='store',
                        dest='LC',
                        choices=["prior"],
                        required=True,
                        help='Local classifier.')

    parser.add_argument('-RC', action='store',
                        dest='RC',
                        choices=["nBC"],
                        required=True,
                        help='Relational classifier.')

    parser.add_argument('-CI', action='store',
                        dest='CI',
                        choices=["relaxation"],
                        required=True,
                        help='Collective inference algorithm,')

    parser.add_argument('-sampling', action='store',
                        dest='sampling',
                        required=True,
                        choices=["nodes", "nedges", "degree", "neighbors", "partial_crawls"],
                        help='Sampling method (nodes, nedges, degree, neighbors, partial_crawls).',
                        )

    parser.add_argument('-pseeds', action='store',
                        dest='pseeds',
                        type=float,
                        required=True,
                        help='Fraction of seed nodes (0, ... ,1).')

    parser.add_argument('-epoch', action='store',
                        dest='epoch',
                        type=int,
                        required=True,
                        help='Epoch or # iteration (1 to 10).')

    parser.add_argument('-ignoreInt', action='store',
                        dest='ignoreInt',
                        type=int,
                        help='Class value to ignore (as int).')

    parser.add_argument('-sn', action='store',
                        dest='sn',
                        type=float,
                        default=None,
                        help='Super node size (partial crawls algo).')

    parser.add_argument('-output', action='store',
                        dest='output',
                        required=True,
                        help='Directory to store all results.')

    parser.add_argument('--version', action='version', version='%(prog)s 1.0')

    results = parser.parse_args()

    print("===================================================")
    print("= ARGUMENTS PASSED:                               =")
    print("===================================================")
    print('datafn ..... = ', results.datafn)
    print('LC ......... = ', results.LC)
    print('RC ......... = ', results.RC)
    print('CI ......... = ', results.CI)
    print('sampling ... = ', results.sampling)
    print('pseeds ..... = ', results.pseeds)
    print('sn ......... = ', results.sn)
    print('epoch ...... = ', results.epoch)
    print('ignoreInt .. = ', results.ignoreInt)
    print('output ..... = ', results.output)
    print("===================================================")

    return results
