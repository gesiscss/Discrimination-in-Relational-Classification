from org.gesis.network.network import Network
from org.gesis.inference.inference import Inference
from org.gesis.local.local import Local
from org.gesis.relational.relational import Relational
from org.gesis.sampling.sampling import Sampling
from utils import estimator
from utils.io import printf
from utils.arguments import init_batch_create_network

def run(params):

    ### 1. Load empirical network
    if params.fit is not None:
        print("")
        printf("*** Loading empirical Network ***")
        emp = Network()
        emp.load(datafn=params.fit, ignoreInt=params.ignoreInt)
        N = emp.G.number_of_nodes()
        m = max(estimator.get_min_degree(emp.G), 2)
        H = round(estimator.get_homophily(emp.G), 1)
        B = round(estimator.get_minority_fraction(emp.G), 1)
        fit = emp.G.graph['name']

    else:
        N = params.N
        m = params.m
        H = params.H
        B = params.B
        fit = None

    ### 2. Create network using model
    print("")
    printf("*** Creating Network ***")
    net = Network(params.kind, fit)
    net.create_network(N=N, m=m, B=B, H=H, i=params.i, x=params.x)
    net.info()
    net.save(params.root)

    printf("done!")

if __name__ == "__main__":
    params = init_batch_create_network()
    run(params)