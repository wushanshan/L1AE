"""Run l1_min in parallel for rcv1 dataset"""
from __future__ import division
from time import time
from joblib import Parallel, delayed
from utils import l1_min_avg_err
from baselines import l1_min, PCA_l1
from sklearn.datasets import fetch_rcv1
from sklearn.model_selection import train_test_split

import numpy as np

SEED = 43
np.random.seed(SEED)


def merge_dict(a, b):
    """Merge two dictionaries"""
    for k in b.keys():
        if k in a:
            a[k].append(b[k])
        else:
            a[k] = [b[k]]


def average_dicts(d):
    """d is a list of dictionaries"""
    for k in d[0].keys():
        d[0][k] = np.mean([d[i][k] for i in xrange(len(d))])
    return d[0]


def evaluate_encoder_matrix(A, Y, X):
    """
    Run l1-min and model-based CoSaMP using the sensing matrix A.
    Args:
        A: 2-D array, shape=(emb_dim, input_dim)
        Y: 2-D array, shape=(num_sample, emb_dim)
        X: 2-D csr_matrix, shape=(num_sample, input_dim)
        feature_indices: 1-D array, defines the onehot-sparsity model.
    """
    l1ae_l1_err, l1ae_l1_exact, _ = l1_min_avg_err(A, Y,
                                                   X, use_pos=False)
    l1ae_l1_err_pos, l1ae_l1_exact_pos, _ = l1_min_avg_err(
                                                       A, Y,
                                                       X, use_pos=True)
    res = {}
    res['l1ae_l1_err'] = l1ae_l1_err
    res['l1ae_l1_exact'] = l1ae_l1_exact
    res['l1ae_l1_err_pos'] = l1ae_l1_err_pos
    res['l1ae_l1_exact_pos'] = l1ae_l1_exact_pos
    return res


# fetch the dataset
X = fetch_rcv1(subset='train').data
# split into train/valid/test
X_train, X_valid = train_test_split(X, test_size=0.4, random_state=SEED)
X_test, X_valid = train_test_split(X_valid, test_size=0.5, random_state=SEED)
input_dim = X_train.shape[1]

# model parameters
emb_dims = [200, 400, 600, 800, 1000]
decoder_num_steps = 10
num_experiment = 1
checkpoint_dir = 'ckpts/rcv1/'

# parallel computation parameters
num_core = 5  # number of cores in your computer
# change batch to a small number (e.g., 20) if runs too slow
batch = int(X_test.shape[0]/num_core)

for emb_dim in emb_dims:
    results_dict = {}
    file_name = ('results_test_20_depth_%d_'+'emb_%d.npy') % (
                                                            decoder_num_steps,
                                                            emb_dim)
    file_path = checkpoint_dir + file_name
    res = np.load(file_path).item()
    results_dict['l1ae_err'] = res['l1ae_err']
    for experiment_i in xrange(num_experiment):
        print("Emb_dim %d, Experiment_%d......") % (emb_dim, experiment_i)
        file_name = ('encoder_weight_depth_%d_'+'emb_%d_'+'experi_%d.npy') % (
                                                  decoder_num_steps,
                                                  emb_dim,
                                                  experiment_i)
        file_path = checkpoint_dir + file_name
        # run l1-min and model-based CoSaMP using the learned sensing matrix
        print("Run l1_min and CoSaMP using learned sensing matrix......")
        t0 = time()
        G = np.load(file_path)
        res = Parallel(n_jobs=num_core)(delayed(evaluate_encoder_matrix)(
                       np.transpose(G), X_test[i*batch:(i+1)*batch, :].dot(G),
                       X_test[i*batch:(i+1)*batch, :])
                       for i in xrange(num_core))
        avg_res = average_dicts(res)
        t1 = time()
        print("Evaluation using the learned matrix takes %f secs") % (t1-t0)
        merge_dict(results_dict, avg_res)
        # l1 minimization
        print("Start parallel l1-min......")
        t0 = time()
        res = Parallel(n_jobs=num_core)(delayed(l1_min)(
                                   X_test[i*batch:(i+1)*batch, :],
                                   input_dim,
                                   emb_dim) for i in xrange(num_core))
        avg_res = average_dicts(res)
        t1 = time()
        print("Parallel l1-min takes %f secs") % (t1-t0)
        merge_dict(results_dict, avg_res)
        # PCA
        print("Start PCA and parallel l1-min......")
        t0 = time()
        res = Parallel(n_jobs=num_core)(delayed(PCA_l1)(
                                     X_train, X_test[i*batch:(i+1)*batch, :],
                                     input_dim, emb_dim)
                                     for i in xrange(num_core))
        avg_res = average_dicts(res)
        t1 = time()
        print("PCA and parallel l1-min takes %f secs") % (t1-t0)
        merge_dict(results_dict, avg_res)
    num_test_sample = num_core*batch
    file_name = ('results_test_%d_depth_%d_'+'emb_%d.npy') % (
                                                            num_test_sample,
                                                            decoder_num_steps,
                                                            emb_dim)
    file_path = checkpoint_dir + file_name
    np.save(file_path, results_dict)
