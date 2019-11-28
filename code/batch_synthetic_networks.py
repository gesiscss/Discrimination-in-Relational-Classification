import sys
from utils.arguments import init_batch
from org.gesis.network.network import Network
from org.gesis.sampling.sampling import Sampling
from org.gesis.local.local import Local
from org.gesis.relational.relational import Relational
from org.gesis.inference.inference import Inference

def run(params):
    ### 1. Input network
    print("\n*** Network ***")
    net = Network()
    net.load(params.datafn)
    net.info()

    ### 2. Sampling
    print("\n*** Sample ***")
    sam = Sampling(params.sampling, net.G, params.pseeds)
    sam.extract_subgraph()
    sam.info()

    ### 3. Local Modeling
    print("\n*** Local model ***")
    local_model = Local(params.LC)
    local_model.learn(sam.Gseeds)
    local_model.info()

    ### 4. Relational Modeling
    print("\n*** Relational model ***")
    relational_model = Relational(params.RC).get_model()
    relational_model.learn(sam.Gseeds)
    relational_model.info()

    ### 5. Inference
    print("\n*** Inference ***")
    inference = Inference(params.CI)
    inference.predict(net.G, local_model, relational_model)
    inference.evaluation()
    inference.summary()
    inference.save(params.output, params.epoch)

    print("done!")

if __name__ == "__main__":
    params = init_batch()
    run(params)