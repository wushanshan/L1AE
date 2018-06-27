"""Run l1_min in parallel for amazon dataset"""
from __future__ import division
from time import time
from joblib import Parallel, delayed
from datasets import amazon_onehot_sparse_data
from utils import l1_min_avg_err, CoSaMP_onehot_avg_err
from baselines import onehot_sparse_CoSaMP, l1_min, PCA_l1

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


def evaluate_encoder_matrix(A, Y, X, feature_indices):
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
    l1ae_cosamp_err, l1ae_cosamp_exact, _ = CoSaMP_onehot_avg_err(
                                                       A, Y,
                                                       X, feature_indices,
                                                       use_pos=False)
    l1ae_cosamp_err_pos, l1ae_cosamp_exact_pos, _ = CoSaMP_onehot_avg_err(
                                                       A, Y,
                                                       X, feature_indices,
                                                       use_pos=True)
    res = {}
    res['l1ae_l1_err'] = l1ae_l1_err
    res['l1ae_l1_exact'] = l1ae_l1_exact
    res['l1ae_l1_err_pos'] = l1ae_l1_err_pos
    res['l1ae_l1_exact_pos'] = l1ae_l1_exact_pos
    res['l1ae_cosamp_err'] = l1ae_cosamp_err
    res['l1ae_cosamp_exact'] = l1ae_cosamp_exact
    res['l1ae_cosamp_err_pos'] = l1ae_cosamp_err_pos
    res['l1ae_cosamp_exact_pos'] = l1ae_cosamp_exact_pos
    return res


# model parameters
emb_dims = [20, 40, 60, 80, 100]
decoder_num_steps = 60
num_experiment = 1
data_dir = 'AmazonAccess/train.csv'
checkpoint_dir = 'ckpts/amazon/'

# load the dataset
X_train, X_valid, X_test, feature_indices = amazon_onehot_sparse_data(
                                                                data_dir,
                                                                SEED=SEED)
input_dim = X_train.shape[1]

# parallel computation parameters
num_core = 50  # number of cores in your computer
# change batch to a small number (e.g., 20) if runs too slow
batch = int(X_test.shape[0]/num_core)

for emb_dim in emb_dims:
    results_dict = {}
    file_name = ('results_test_100_depth_%d_'+'emb_%d.npy') % (
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
                       X_test[i*batch:(i+1)*batch, :], feature_indices)
                       for i in xrange(num_core))
        avg_res = average_dicts(res)
        t1 = time()
        print("Evaluation using the learned matrix takes %f secs") % (t1-t0)
        merge_dict(results_dict, avg_res)
        # model based CoSaMP
        print("Start model-based CoSaMP......")
        t0 = time()
        res = onehot_sparse_CoSaMP(X_test, input_dim, emb_dim, feature_indices)
        t1 = time()
        print("Model-based CoSaMP takes %f secs") % (t1-t0)
        merge_dict(results_dict, res)
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
