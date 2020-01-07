from org.gesis.inference import inference
from org.gesis.inference.inference import Inference
from utils import estimator
from utils.arguments import init_batch_summary
from utils.io import printf
from utils.io import write_csv

def run(params):

    if not params.overwrite and inference.is_inference_summary_done(params.output, params.kind, params.LC, params.RC, params.CI, params.sampling):
        printf("Already done!")
        return

    ### 1. Load results
    print("")
    printf("*** Loading evaluation files ***")
    df_results = Inference.get_all_results_as_dataframe(params.output,
                                                        kind=params.kind,
                                                        LC=params.LC,
                                                        RC=params.RC,
                                                        CI=params.CI,
                                                        sampling=params.sampling,
                                                        njobs=params.njobs,
                                                        verbose=True)
    print(df_results.head(5))
    print(df_results.shape)
    print(df_results.sampling.unique())
    print()

    ### 2. Global estimates
    print("")
    printf("*** Updating global estimates ***")
    df_results = estimator.merge_global_estimates(df_results, params.LC, params.RC,
                                                  params.output,
                                                  njobs=params.njobs)
    print(df_results.head(5))
    print(df_results.shape)
    print()

    ### 3. saving
    print("")
    printf("*** Saving ***")
    fn = inference.get_inference_summary_fn(params.output, params.kind, params.LC, params.RC, params.CI, params.sampling)
    write_csv(df_results, fn)

    print(df_results.sample(5))
    print(df_results.shape)
    print(df_results.sampling.unique())
    print()

    printf("done!")

if __name__ == "__main__":
    params = init_batch_summary()
    run(params)