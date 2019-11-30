import os
import networkx as nx
import pickle
import datetime

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

def printf(txt):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("{}:\t{}".format(now, txt))