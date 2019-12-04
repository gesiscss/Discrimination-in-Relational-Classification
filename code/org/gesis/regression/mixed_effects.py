############################################
# System dependencies
############################################
import os

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from patsy import dmatrices
############################################
# Local dependencies
############################################
from utils.io import write_pickle


############################################
# Fucntions
############################################
def is_regression_done(output, kind, sampling, groups):
    return

############################################
# Class
############################################
class MixedEffects(object):

    def __init__(self, dependent_var, predictors, groups, all_attributes, df_results=None):
        self.groups = groups
        self.dependent_var = dependent_var
        self.predictors = predictors
        self.all_attributes = all_attributes
        self.df_results = df_results

        self.group_vars = self._get_attributes()
        self.formula = None
        self.fe_params = None
        self.random_effects = None
        self.df_test = None
        self.mdf = None
        self.X = None
        self.Y = None
        self.Z = None

    def _get_attributes(self):
        vars = []
        for ch in self.groups:
            vars.append('pseeds' if ch == 'p' else ch if ch in self.all_attributes else None)
        return vars

    def modeling(self):
        self.make_groups()
        self.compute_mean_values()
        self.fit()

    def make_groups(self):
        self.df_results.loc[:, self.groups] = self.df_results.apply(lambda row: "_".join([str(row[v]) for v in self.group_vars]), axis=1)

    def compute_mean_values(self):
        condi_sel = [self.groups] + self.all_attributes
        self.df_test = pd.DataFrame({'Nt': self.df_results.groupby(condi_sel).size()}).reset_index()
        self.df_test[self.groups] = self.df_test[self.groups].astype('category')
        self.df_test[self.dependent_var] = np.asarray(self.df_results.groupby(condi_sel)[self.dependent_var].mean())

    def fit(self):
        self.formula = "{} ~ {}".format(self.dependent_var, ' + '.join(self.predictors))
        md = smf.mixedlm(self.formula, self.df_test, groups=self.df_test[self.groups])
        self.mdf = md.fit()

        # storing results in dataframe
        self.fe_params = pd.DataFrame(self.mdf.fe_params, columns=['LMM'])
        self.random_effects = pd.DataFrame(self.mdf.random_effects)
        self.random_effects = self.random_effects.transpose()
        self.random_effects = self.random_effects.rename(index=str, columns={'Group': 'LMM'})

        #storing results in design matrix
        self.Y, self.X = dmatrices(self.formula, data=self.df_test, return_type='matrix')
        _, self.Z = dmatrices('{} ~ -1+{}'.format(self.dependent_var, self.groups), data=self.df_test, return_type='matrix')
        self.X = np.asarray(self.X)  # fixed effect
        self.Z = np.asarray(self.Z)  # mixed effect
        self.Y = np.asarray(self.Y).flatten()


    def summary(self):
        print(self.mdf.summary())

    def save_summary_as_latex(self, path):
        latex = self.mdf.summary().as_latex()

        if path is not None:
            fn = os.path.join(path, "fixed_effects_summary.latex")

            with open(fn, 'w') as f:
                f.write(latex)
                f.write('\n')
            print('{} saved!'.format(fn))

    def save(self, path):

        fn = os.path.join(path, 'fe_params.pickle')
        write_pickle(self.fe_params, fn)

        fn = os.path.join(path, 'random_effects.pickle')
        write_pickle(self.random_effects, fn)

        fn = os.path.join(path, 'X.pickle')
        write_pickle(self.X, fn)

        fn = os.path.join(path, 'Y.pickle')
        write_pickle(self.Y, fn)

        fn = os.path.join(path, 'Z.pickle')
        write_pickle(self.Z, fn)


