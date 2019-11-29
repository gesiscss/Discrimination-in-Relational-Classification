import argparse


def init_batch():
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

    parser.add_argument('-ignore', action='store',
                        dest='ignore',
                        help='Class value to ignore.')

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
    print('epoch ...... = ', results.epoch)
    print('ignore ..... = ', results.ignore)
    print('output ..... = ', results.output)
    print("===================================================")

    return results