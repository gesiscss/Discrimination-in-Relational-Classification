import datetime
import os
import pickle

import networkx as nx
import numpy as np
import pandas as pd


def create_folder(path, changeto=False):
    if not os.path.exists(path):
        print("{} creating...".format(path))
        try:
            os.makedirs(path)
            print("{} created!".format(path))
        except Exception as ex:
            print(ex)
    if changeto:
        try:
            os.chdir(path)
        except:
            pass

def load_pickle(fn, verbose=True):
    try:
        with open(fn, 'rb') as f:
            obj = pickle.load(f)

            if verbose:
                print("{} loaded!".format(fn))
            return obj
    except Exception as ex:
        print(ex)

def write_pickle(obj, fn):
    try:
        with open(fn, 'wb') as f:
            pickle.dump(obj, f)
            print("{} saved!".format(fn))
    except Exception as ex:
        print(ex)

def write_gpickle(graph, fn):
    try:
        if not fn.endswith('.gpickle'):
            fn = '{}.gpickle'.format(fn)

        nx.write_gpickle(graph, fn)
        print("{} saved!".format(fn))
    except Exception as ex:
        print(ex)

def load_gpickle(datafn):
    if os.path.exists(datafn):
        try:
            graph = nx.read_gpickle(datafn)
        except Exception as ex:
            print(ex)
    else:
        raise FileNotFoundError("{} does not exist.".format(datafn))
    return graph

def write_csv(df, fn):
    try:
        df.to_csv(fn, index=False)
        print("{} saved!".format(fn))
    except Exception as ex:
        print(ex)


def load_csv(fn):
    if os.path.exists(fn):
        try:
            df = pd.read_csv(fn, index_col=False)
        except Exception as ex:
            print(ex)
    else:
        raise FileNotFoundError("{} does not exist.".format(fn))
    return df

def get_random_datafn(path,kind,N,m,B,H):
    # BAH-N500-m4-B0.5-H1.0-i5-x5-h1.0-k7.9-km7.9-kM7.9.gpickle
    fn = [fn for fn in os.listdir(path)
          if fn.startswith(kind) and
          fn.endswith('.gpickle') and
          '-N{}-'.format(N) in fn and
          '-m{}-'.format(m) in fn and
          '-B{}-'.format(B) in fn and
          '-H{}-'.format(H) in fn]

    if len(fn) == 0:
        raise Exception('No files found.')

    return os.path.join(path, np.random.choice(fn,1)[0])

def printf(txt):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("{}:\t{}".format(now, txt))