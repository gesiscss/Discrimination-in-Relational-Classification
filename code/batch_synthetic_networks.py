from utils.arguments import init_batch
from org.gesis.network.network import Network
from org.gesis.sampling.sampling import Sampling
from org.gesis.local.local import Local
from org.gesis.relational.relational import Relational
from org.gesis.inference.inference import Inference
from org.gesis.inference.inference import is_inference_done
from utils.io import printf

def run(params):

    if is_inference_done(params.output, params.datafn, params.pseeds, params.epoch):
        printf("Already done!")
        return

    ### 1. Input network
    printf("\n*** Network ***")
    net = Network()
    net.load(params.datafn, params.ignoreInt)
    net.info()

    ### 2. Sampling
    printf("\n*** Sample ***")
    sam = Sampling(params.sampling, net.G, params.pseeds)
    sam.extract_subgraph()
    sam.info()

    ### 3. Local Modeling
    printf("\n*** Local model ***")
    local_model = Local(params.LC)
    local_model.learn(sam.Gseeds)
    local_model.info()

    ### 4. Relational Modeling
    printf("\n*** Relational model ***")
    relational_model = Relational(params.RC).get_model()
    relational_model.learn(sam.Gseeds)
    relational_model.info()

    ### 5. Inference
    printf("\n*** Inference ***")
    inference = Inference(params.CI)
    inference.predict(net.G, local_model, relational_model)
    inference.evaluation()
    inference.summary()
    inference.save(params.output, params.epoch)

    printf("done!")

if __name__ == "__main__":
    params = init_batch()
    run(params)