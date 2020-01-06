import os
import networkx as nx
import pickle
import datetime
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

def printf(txt):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("{}:\t{}".format(now, txt))