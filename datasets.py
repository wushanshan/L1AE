# Copyright 2018 Shanshan Wu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Prepare the datasets"""
from __future__ import division
from scipy import sparse
from sklearn.preprocessing import normalize
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

import numpy as np


def synthetic_power_law_data(input_dim, powerlaw_exp, powerlaw_bias,
                             avg_sparsity, num_samples,
                             train_ratio=0.6, valid_ratio=0.2):
    """
    Generate synthetic sparse dataset with average sparsity = avg_sparsity.
    For vector x, its i-th entry is nonzero with a probability proportional to
    1/(i+powerlaw_bias)^(powerlaw_exp), where i is from 0 to input_dim-1.
    """
    probs = 1.0/np.power(np.arange(input_dim)+powerlaw_bias, powerlaw_exp)
    probs = probs/np.sum(probs)*avg_sparsity
    probs = np.minimum(probs, np.ones_like(probs))
    X = np.zeros((num_samples, input_dim))
    for i in xrange(num_samples):
        X[i, :] = np.random.binomial(1, probs)
    X = np.array(X)
    shuffle = np.random.permutation(X.shape[0])
    X = X[shuffle, :]
    # convert it to csr_matrix format
    X = sparse.csr_matrix(X)
    # set the nonzeros to be unifrom in [0,1]
    X.data = np.random.uniform(0.0, 1.0, len(X.data))
    # make each sample unit norm
    normalize(X, norm='l2', axis=1, copy=False, return_norm=False)
    # split into train/valid/test
    train_size = int(train_ratio*num_samples)
    valid_size = int(valid_ratio*num_samples)
    X_train = X[:train_size, :]
    X_valid = X[train_size:(train_size+valid_size), :]
    X_test = X[(train_size+valid_size):, :]
    return X_train, X_valid, X_test


def synthetic_block_sparse_data(input_dim, block_dim,
                                sparsity_level, num_samples,
                                train_ratio=0.6, valid_ratio=0.2):
    """
    Generate synthetic sparse dataset with block sparsity.
    Block-sparsity signals are defined in Section VI of
    the paper "Model-based compressive sensing".
    Each vector has sparsity_level*block_dim nonzeros.
    """
    X = np.zeros((num_samples, input_dim))
    for i in xrange(num_samples):
        # Each vector contains sparsity_level blocks (randomly selected).
        indices = np.random.choice(input_dim//block_dim,
                                   sparsity_level, replace=False)
        for idx in indices:
            X[i, idx*block_dim:(idx+1)*block_dim] = np.random.uniform(
                                                        0.0, 1.0, block_dim)
    X = np.array(X)
    shuffle = np.random.permutation(X.shape[0])
    X = X[shuffle, :]
    # convert it to csr_matrix format
    X = sparse.csr_matrix(X)
    # make each sample unit norm
    normalize(X, norm='l2', axis=1, copy=False, return_norm=False)
    # split into train/valid/test
    train_size = int(train_ratio*num_samples)
    valid_size = int(valid_ratio*num_samples)
    X_train = X[:train_size, :]
    X_valid = X[train_size:(train_size+valid_size), :]
    X_test = X[(train_size+valid_size):, :]
    return X_train, X_valid, X_test


def amazon_onehot_sparse_data(data_dir, SEED=43):
    # load data from the file
    X = np.loadtxt(open(data_dir), delimiter=',',
                   usecols=range(1, 10), skiprows=1)
    # perform one-hot encoding
    encoder = OneHotEncoder()
    encoder.fit(X)
    X = encoder.transform(X)  # returns a csr_matrix
    # make each sample unit norm
    normalize(X, norm='l2', axis=1, copy=False, return_norm=False)
    # split into train/valid/test
    X_train, X_valid = train_test_split(X, test_size=0.4, random_state=SEED)
    X_test, X_valid = train_test_split(X_valid, test_size=0.5,
                                       random_state=SEED)
    # compute the number of active features in each feature range
    feat_range = encoder.feature_indices_
    active_feats = sorted(encoder.active_features_)
    feat_num = np.zeros(len(feat_range), dtype=np.int)
    idx = 1
    for active_feat_idx in active_feats:
        if active_feat_idx < feat_range[idx]:
            feat_num[idx] += 1
        else:
            idx += 1
            feat_num[idx] += 1
    feature_indices = np.cumsum(feat_num)
    return X_train, X_valid, X_test, feature_indices
