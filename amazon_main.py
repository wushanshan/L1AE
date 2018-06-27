"""Train an autoencoder over the one-hot encoded AmazonAccess dataset"""
from __future__ import division
from time import time
from model import L1AE
from datasets import amazon_onehot_sparse_data
from utils import l1_min_avg_err, CoSaMP_onehot_avg_err
from baselines import onehot_sparse_CoSaMP, l1_min, PCA_l1

import os
import numpy as np
import tensorflow as tf

SEED = 43
np.random.seed(SEED)

flags = tf.app.flags
flags.DEFINE_integer("emb_dim", 20, "Number of measurements [20]")
flags.DEFINE_integer("decoder_num_steps", 60,
                     "Depth of the decoder network [60]")
flags.DEFINE_integer("batch_size", 256, "Batch size [256]")
flags.DEFINE_float("learning_rate", 0.01, "Learning rate for SGD [0.01]")
flags.DEFINE_integer("max_training_epochs", 2e4,
                     "Maximum number of training epochs [2e4]")
flags.DEFINE_integer("display_interval", 1,
                     "Print the training info every [1] epochs")
flags.DEFINE_integer("validation_interval", 1,
                     "Compute validation loss every [1] epochs")
flags.DEFINE_integer("max_steps_not_improve", 1,
                     "stop training when the validation loss \
                      does not improve for [5] validation_intervals")
flags.DEFINE_string("checkpoint_dir", "ckpts/amazon/",
                    "Directory name to save the checkpoints \
                    [ckpts/amazon/]")
flags.DEFINE_string("data_dir", "../../AmazonAccess/train.csv",
                    "Directory name to the AmazonAccess dataset \
                    [../../AmazonAccess/train.csv]")
flags.DEFINE_integer("num_experiment", 1,
                     "Number of experiments [1]")

FLAGS = flags.FLAGS


# model parameters
emb_dim = FLAGS.emb_dim
decoder_num_steps = FLAGS.decoder_num_steps

# training parameters
batch_size = FLAGS.batch_size
learning_rate = FLAGS.learning_rate
max_training_epochs = FLAGS.max_training_epochs
display_interval = FLAGS.display_interval
validation_interval = FLAGS.validation_interval
max_steps_not_improve = FLAGS.max_steps_not_improve

# checkpoint directory
checkpoint_dir = FLAGS.checkpoint_dir
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# dataset directory
data_dir = FLAGS.data_dir

# number of experiments
num_experiment = FLAGS.num_experiment


def merge_dict(a, b):
    """Merge two dictionaries"""
    for k in b.keys():
        if k in a:
            a[k].append(b[k])
        else:
            a[k] = [b[k]]


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


X_train, X_valid, X_test_all, feature_indices = amazon_onehot_sparse_data(
                                                                data_dir,
                                                                SEED=SEED)
"""
Note that solving 'l1_min' takes about 3 secs/sample, and the whole test set
contains about 6k samples. For the purpose of saving time, in this script
we only evaluate the first 100 samples of the test set. We still use the
complete training dataset for training and save the 'encoder_weight' to file.
To evaluate over the whole test set, we used a multi-core machine
(more than 30 cores) and ran 'l1_min' in parallel:
######
from joblib import Parallel, delayed
Parallel(n_jobs=30)(delayed(l1_min)(
        X_test[i*210:(i+1)*210,:], input_dim, emb_dim) for i in xrange(30))
######
"""
X_test = X_test_all[:100, :]

results_dict = {}  # dictionary that saves all results

input_dim = X_train.shape[1]

for experiment_i in xrange(num_experiment):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    print("---Experiment: %d---" % (experiment_i))
    sparse_AE = L1AE(sess, input_dim, emb_dim, decoder_num_steps)

    print("Start training......")
    sparse_AE.train(X_train, X_valid, batch_size, learning_rate,
                    max_training_epochs, display_interval,
                    validation_interval, max_steps_not_improve)
    # evaluate the autoencoder
    test_sq_loss = sparse_AE.inference(X_test_all, batch_size)
    # save the encoder_weight
    G = sparse_AE.sess.run(sparse_AE.encoder_weight)
    file_name = ('encoder_weight_depth_%d_'+'emb_%d_'+'experi_%d.npy') % (
                                              decoder_num_steps,
                                              emb_dim,
                                              experiment_i)
    file_path = checkpoint_dir + file_name
    np.save(file_path, G)
    # run l1-min and model-based CoSaMP using the learned sensing matrix
    Y = X_test.dot(G)
    res = evaluate_encoder_matrix(np.transpose(G), Y, X_test, feature_indices)
    res['l1ae_err'] = np.sqrt(test_sq_loss)  # RMSE
    merge_dict(results_dict, res)
    # model based CoSaMP
    print("Start model-based CoSaMP......")
    t0 = time()
    res = onehot_sparse_CoSaMP(X_test, input_dim, emb_dim, feature_indices)
    t1 = time()
    print("Model-based CoSaMP takes %f secs") % (t1-t0)
    merge_dict(results_dict, res)
    # l1 minimization
    print("Start l1-min......")
    t0 = time()
    res = l1_min(X_test, input_dim, emb_dim)
    t1 = time()
    print("L1-min takes %f secs") % (t1-t0)
    merge_dict(results_dict, res)
    # PCA
    print("Start PCA and l1-min......")
    t0 = time()
    res = PCA_l1(X_train, X_test, input_dim, emb_dim)
    t1 = time()
    print("PCA and l1-min takes %f secs") % (t1-t0)
    merge_dict(results_dict, res)

# save results_dict
file_name = ('results_test_100_depth_%d_'+'emb_%d.npy') % (decoder_num_steps,
                                                           emb_dim)
file_path = checkpoint_dir + file_name
np.save(file_path, results_dict)
