import os
import networkx as nx
import pickle

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