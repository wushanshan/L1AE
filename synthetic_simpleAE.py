"""Train an autoencoder over synthetic datasets"""
from __future__ import division
from datasets import synthetic_block_sparse_data
from utils import l1_min_avg_err
from baselines import simple_AE_l1

import os
import numpy as np
import tensorflow as tf

SEED = 43
np.random.seed(SEED)

flags = tf.app.flags
flags.DEFINE_integer("input_dim", 1000, "Input dimension [1000]")
flags.DEFINE_integer("block_dim", 10, "Size of a block in block-sparsity [10]")
flags.DEFINE_integer("sparsity_level", 1, "Sparsity in block-sparsity [1]")
flags.DEFINE_integer("emb_dim", 10, "Number of measurements [10]")
flags.DEFINE_integer("num_samples", 10000, "Number of total samples [10000]")
flags.DEFINE_integer("batch_size", 128, "Batch size [128]")
flags.DEFINE_float("learning_rate", 0.1, "Learning rate for SGD [0.01]")
flags.DEFINE_integer("max_training_epochs", 2e4,
                     "Maximum number of training epochs [2e4]")
flags.DEFINE_integer("display_interval", 100,
                     "Print the training info every [100] epochs")
flags.DEFINE_integer("validation_interval", 1,
                     "Compute validation loss every [10] epochs")
flags.DEFINE_integer("max_steps_not_improve", 1,
                     "stop training when the validation loss \
                      does not improve for [5] validation_intervals")
flags.DEFINE_string("checkpoint_dir", "ckpts/synthetic/",
                    "Directory name to save the checkpoints \
                    [ckpts/synthetic/]")
flags.DEFINE_integer("num_random_dataset", 10,
                     "Number of random datasets [10]")
flags.DEFINE_integer("num_experiment", 1,
                     "Number of experiments for each dataset [1]")

FLAGS = flags.FLAGS


# model parameters
input_dim = FLAGS.input_dim
block_dim = FLAGS.block_dim
sparsity_level = FLAGS.sparsity_level
emb_dim = FLAGS.emb_dim
num_samples = FLAGS.num_samples

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

results_dict = {}


def merge_dict(a, b):
    """Merge two dictionaries"""
    for k in b.keys():
        if k in a:
            a[k].append(b[k])
        else:
            a[k] = [b[k]]


for dataset_i in xrange(num_random_dataset):
    X_train, X_valid, X_test = synthetic_block_sparse_data(
                                                input_dim, block_dim,
                                                sparsity_level, num_samples)
    for experiment_i in xrange(num_experiment):
        print("---Dataset: %d, Experiment: %d---" % (dataset_i, experiment_i))
        err, G = simple_AE_l1(input_dim, emb_dim, X_train, X_valid, X_test,
                              batch_size, learning_rate, max_training_epochs,
                              display_interval, validation_interval,
                              max_steps_not_improve)
        Y = X_test.dot(G)
        sim_l1_err, sim_l1_exact, _ = l1_min_avg_err(np.transpose(G), Y,
                                                     X_test, use_pos=False)
        sim_l1_err_pos, sim_l1_exact_pos, _ = l1_min_avg_err(
                                                       np.transpose(G), Y,
                                                       X_test, use_pos=True)

        res = {}
        res['simple_ae_err'] = np.sqrt(err)  # RMSE
        res['simple_ae_l1_err'] = sim_l1_err
        res['simple_ae_l1_exact'] = sim_l1_exact
        res['simple_ae_l1_err_pos'] = sim_l1_err_pos
        res['simple_ae_l1_exact_pos'] = sim_l1_exact_pos
        merge_dict(results_dict, res)

# save results_dict
file_name = ('simple_ae_input_%d_'+'block_%d_'+'sparse_%d_') % (input_dim,
                                                                block_dim,
                                                                sparsity_level)
file_name = file_name + ('emb_%d.npy') % (emb_dim)
file_path = checkpoint_dir + file_name

np.save(file_path, results_dict)
