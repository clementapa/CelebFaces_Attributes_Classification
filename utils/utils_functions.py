import os, errno
import numpy as np

def create_dir(dir_name):
    try:
        os.makedirs(dir_name)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def balace_input(X, y, attribute_idx):
    pos_idx, neg_idx = np.where(y[:, attribute_idx]==1)[0], np.where(y[:, attribute_idx]==0)[0]
    num_samples = min(len(pos_idx), len(neg_idx))
    if len(pos_idx) >= len(neg_idx):
        positive_samples = X[pos_idx][:num_samples], y[pos_idx, attribute_idx][:num_samples]
        negative_samples = X[neg_idx], y[neg_idx, attribute_idx]
    else:
        positive_samples = X[pos_idx], y[pos_idx, attribute_idx]
        negative_samples = X[neg_idx][:num_samples], y[neg_idx, attribute_idx][:num_samples]
    return positive_samples, negative_samples