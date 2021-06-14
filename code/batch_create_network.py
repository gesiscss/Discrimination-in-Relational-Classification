from org.gesis.network.network import Network
from org.gesis.inference.inference import Inference
from org.gesis.local.local import Local
from org.gesis.relational.relational import Relational
from org.gesis.sampling.sampling import Sampling
from utils import estimator
from utils.io import printf
from utils.io import load_csv
from utils.arguments import init_batch_create_network
import numpy as np

def run(params):

    ### 1. Load empirical network (asymmetric homophily)
    if params.fit is not None:
        print("")
        printf("*** Loading empirical Network ***")
        emp = Network()
        emp.load(datafn=params.fit, ignoreInt=params.ignoreInt)
        #N = emp.G.number_of_nodes() if params.N is None else params.N
        density = estimator.get_density(emp.G) if params.density is None else params.density
        N, E = estimator.get_expected_N_E(density)
        m = max(estimator.get_min_degree(emp.G), 2)
        B = round(estimator.get_minority_fraction(emp.G), 2)
        Emm, EmM, EMM, EMm = estimator.get_edge_type_counts(emp.G, True)
        fit = emp.G.graph['name']
        
        if params.evalues is not None:
            df_evalues = load_csv(params.evalues)
            Hmm,HMM = estimator.find_homophily_MLE(emp.G, df_evalues)
            H = round(np.mean([Hmm,HMM]),2)
        else:
            try:
                H = round(emp.G.graph['H'], 2) #None
                Hmm = round(emp.G.graph['Hmm'], 2)
                HMM = round(emp.G.graph['HMM'], 2)
            except:
                H = None
                Hmm = None
                HMM = None
    else:
        N = params.N
        m = params.m
        H = params.H
        Hmm = params.Hmm
        HMM = params.HMM
        B = params.B
        density = params.density
        fit = None

    ### 2. Create network using model
    print("")
    printf("*** Creating Network ***")
    net = Network(params.kind, fit)
    net.create_network(N=N, m=m, B=B, density=density, H=H, Hmm=Hmm, HMM=HMM, i=params.i, x=params.x)
    net.info()
    net.save(params.root)

    printf("done!")

if __name__ == "__main__":
    params = init_batch_create_network()
    run(params)