import numpy as np
def calclualte_low_var_dim(score_norm_refs):
    sn_features = np.concatenate([ref.feature for ref in score_norm_refs], axis=0)
    low_var_dim = sn_features.var(axis=0).argmin()
    return low_var_dim

