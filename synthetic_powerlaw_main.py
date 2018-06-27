"""Train an autoencoder over a synthetic dataset with power-law sparsity"""
from __future__ import division
from time import time
from model import L1AE
from datasets import synthetic_power_law_data
from utils import l1_min_avg_err
from baselines import l1_min, PCA_l1

import os
import numpy as np
import tensorflow as tf

SEED = 43
np.random.seed(SEED)

flags = tf.app.flags
flags.DEFINE_integer("input_dim", 1000, "Input dimension [1000]")
flags.DEFINE_integer("powerlaw_exp", 1, "Exponent in the power law [1]")
flags.DEFINE_integer("powerlaw_bias", 1, "Bias in the power law [1]")
flags.DEFINE_integer("avg_sparsity", 10, "Average nonzeros [10]")
flags.DEFINE_integer("num_samples", 10000, "Number of total samples [10000]")
flags.DEFINE_integer("emb_dim", 10, "Number of measurements [10]")
flags.DEFINE_integer("decoder_num_steps", 10,
                     "Depth of the decoder network [10]")
flags.DEFINE_integer("batch_size", 128, "Batch size [128]")
flags.DEFINE_float("learning_rate", 0.01, "Learning rate for SGD [0.01]")
flags.DEFINE_integer("max_training_epochs", 2e4,
                     "Maximum number of training epochs [1e3]")
flags.DEFINE_integer("display_interval", 100,
                     "Print the training info every [100] epochs")
flags.DEFINE_integer("validation_interval", 10,
                     "Compute validation loss every [10] epochs")
flags.DEFINE_integer("max_steps_not_improve", 5,
                     "stop training when the validation loss \
                      does not improve for [5] validation_intervals")
flags.DEFINE_string("checkpoint_dir", "ckpts/synthetic_powerlaw/",
                    "Directory name to save the checkpoints \
                    [ckpts/synthetic_powerlaw/]")
flags.DEFINE_integer("num_random_dataset", 10,
                     "Number of random datasets [10]")
flags.DEFINE_integer("num_experiment", 1,
                     "Number of experiments [1]")

FLAGS = flags.FLAGS


# model parameters
input_dim = FLAGS.input_dim
powerlaw_exp = FLAGS.powerlaw_exp
powerlaw_bias = FLAGS.powerlaw_bias
avg_sparsity = FLAGS.avg_sparsity
num_samples = FLAGS.num_samples
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

# number of experiments
num_random_dataset = FLAGS.num_random_dataset
num_experiment = FLAGS.num_experiment


def merge_dict(a, b):
    """Merge two dictionaries"""
    for k in b.keys():
        if k in a:
            a[k].append(b[k])
        else:
            a[k] = [b[k]]


def evaluate_encoder_matrix(A, Y, X):
    """
    Run l1-min using the sensing matrix A.
    Args:
        A: 2-D array, shape=(emb_dim, input_dim)
        Y: 2-D array, shape=(num_sample, emb_dim)
        X: 2-D csr_matrix, shape=(num_sample, input_dim)
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


results_dict = {}  # dictionary that saves all results


for dataset_i in xrange(num_random_dataset):
    X_train, X_valid, X_test = synthetic_power_law_data(
                                                input_dim, powerlaw_exp,
                                                powerlaw_bias, avg_sparsity,
                                                num_samples)
    true_sparsity = len(X_train.data)/X_train.shape[0]
    res = {}
    res['true_sparsity'] = true_sparsity
    merge_dict(results_dict, res)
    for experiment_i in xrange(num_experiment):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        print("---Dataset: %d, Experiment: %d---" % (dataset_i, experiment_i))
        sparse_AE = L1AE(sess, input_dim, emb_dim, decoder_num_steps)

        print("Start training......")
        sparse_AE.train(X_train, X_valid, batch_size, learning_rate,
                        max_training_epochs, display_interval,
                        validation_interval, max_steps_not_improve)

        # evaluate the autoencoder
        test_sq_loss = sparse_AE.inference(X_test, batch_size)
        # run l1-min using the learned sensing matrix
        G = sparse_AE.sess.run(sparse_AE.encoder_weight)
        Y = X_test.dot(G)
        res = evaluate_encoder_matrix(np.transpose(G), Y, X_test)
        res['l1ae_err'] = np.sqrt(test_sq_loss)  # RMSE
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
file_name = ('input_%d_'+'powerexp_%d_'+'powerbias_%d_') % (input_dim,
                                                            powerlaw_exp,
                                                            powerlaw_bias)
file_name = file_name + ('sparse_%d_'+'depth_%d_'+'emb_%d.npy') % (
                                                        avg_sparsity,
                                                        decoder_num_steps,
                                                        emb_dim)
file_path = checkpoint_dir + file_name
np.save(file_path, results_dict)
