import os

from org.gesis.inference.inference import Inference
from org.gesis.regression.mixed_effects import MixedEffects
from org.gesis.regression.mixed_effects import is_regression_done
from utils.arguments import init_batch_mixed_effects
from utils.io import create_folder
from utils.io import printf

X_var = "rocauc"
y_vars = ['N','density','B']
ALL_attributes = ['N','density','B','H','pseeds']

def run(params):

    if is_regression_done(params.output, params.kind, params.sampling, params.groups):
        printf("Already done!")
        return

    ### 1. Load results
    df_results = Inference.get_all_results_as_dataframe(params.output, params.kind, sampling=params.sampling, njobs=params.njobs, verbose=params.verbose)
    print("{} records loaded.".format(df_results.shape))

    ### 2. Mixed Effects model
    me = MixedEffects(X_var, y_vars, params.groups, ALL_attributes, df_results)
    me.modeling()

    ### 3. Save & Summary
    output = os.path.join(params.output, 'mixed_effects')
    create_folder(output)

    me.summary()
    me.save_summary_as_latex(output)
    me.save(output)

if __name__ == "__main__":
    params = init_batch_mixed_effects()
    printf("init!")
    run(params)
    printf("done!")