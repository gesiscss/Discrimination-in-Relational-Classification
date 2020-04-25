import pandas as pd
from org.gesis.regression.mixed_effects import MixedEffects


def setup(df_realworld):
    df_realworld.loc[:, 'B'] = df_realworld.B.round(2)
    df_realworld.loc[:, 'H'] = df_realworld.H.round(2)
    df_realworld.loc[:, 'kind'] = "LME"

    df_realworld.loc[df_realworld.query("dataset=='Caltech36'").index, 'H'] = 0.56
    df_realworld.loc[df_realworld.query("dataset=='Swarthmore42'").index, 'H'] = 0.53
    df_realworld.loc[df_realworld.query("dataset=='USF51'").index, 'H'] = 0.45
    df_realworld.loc[df_realworld.query("dataset=='Wikipedia'").index, 'H'] = 0.60
    df_realworld.loc[df_realworld.query("dataset=='Escorts'").index, 'H'] = 0.00

    df_realworld.sort_values(['dataset', 'pseeds', 'epoch'], inplace=True)
    df_realworld.reset_index(inplace=True, drop=True)
    return df_realworld

def predict_allrows(df, mdf, params):

    newdf = pd.DataFrame(columns=df.columns)

    for index, row in df.iterrows():
        m = MixedEffects.predict_one(row, mdf, params['group_vars'])

        newdf = newdf.append({  'kind': 'LME',
                                'dataset': row.dataset,
                                'N': row.N,
                                'm': row.m,
                                'density': row.density,
                                'B': row.B,
                                'H': row.H,
                                'sampling': row.sampling,
                                'pseeds': row.pseeds,
                                'ROCAUC': m,
                                'epoch': row.epoch
                                }, ignore_index=True)[newdf.columns]
    return newdf
