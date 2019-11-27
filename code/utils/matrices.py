import numpy as np

def symmetrize(df):
    return df + df.T - np.diag(np.diag(df))