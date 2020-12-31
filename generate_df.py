import pandas as pd
import numpy as np


def disagreement_df(n_features, block_size, block_proba):
    assert block_size > 0, "Incorrect block size."
    assert block_proba >= 0.0 and block_proba <= 1.0, "Incorrect block probability."
    assert n_features >= 1, "At least one feature for demo."
    multiplier = 10
    n_activated = int(block_size * block_proba)
    zero_observations_block = np.zeros(block_size * multiplier)
    zero_observations_block[:(n_activated * multiplier)] = 1.0
    one_observations_block = np.ones(block_size)
    one_observations_block[:n_activated] = 0.0
    observations = np.concatenate([one_observations_block] + [zero_observations_block for _ in range(n_features - 1)])
    assert observations.shape[0] == (n_features - 1) * multiplier * block_size + block_size
    features = np.identity(n_features)[1:]
    features = np.repeat(features, axis=0, repeats=block_size * multiplier)
    assert features.shape[0] == block_size * multiplier * (n_features - 1)
    features = np.concatenate([np.ones((block_size, n_features)), features], axis=0)
    assert features.shape[0] == (n_features - 1) * multiplier * block_size + block_size
    return features, observations

if __name__ == "__main__":
    print(disagreement_df(n_features=2, block_size=10, block_proba=0.8))