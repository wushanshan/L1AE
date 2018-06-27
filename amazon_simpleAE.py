"""Train an autoencoder over the one-hot encoded AmazonAccess dataset"""
from __future__ import division
from datasets import amazon_onehot_sparse_data
from utils import l1_min_avg_err
from baselines import simple_AE_l1

import os
import numpy as np
import tensorflow as tf

SEED = 43
np.random.seed(SEED)

flags = tf.app.flags
flags.DEFINE_integer("emb_dim", 20, "Number of measurements [20]")
flags.DEFINE_integer("decoder_num_steps", 0,
                     "Depth of the decoder network [10]")
flags.DEFINE_integer("batch_size", 256, "Batch size [256]")
flags.DEFINE_float("learning_rate", 0.5, "Learning rate for SGD [0.01]")
flags.DEFINE_integer("max_training_epochs", 1e4,
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

# training parameters
batch_size = FLAGS.batch_size
learning_rate = FLAGS.learning_rate
max_training_epochs = FLAGS.max_training_epochs
display_interval = FLAGS.display_interval
validation_interval = FLAGS.validation_interval
max_steps_not_improve = FLAGS.max_steps_not_improve
decoder_num_steps = FLAGS.decoder_num_steps


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
    print("---Experiment: %d---" % (experiment_i))
    err, G = simple_AE_l1(input_dim, emb_dim, X_train, X_valid, X_test_all,
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

    file_name = ('simple_ae_amazon_encoder_weight_'+'emb_%d.npy') % (emb_dim)
    file_path = checkpoint_dir + file_name
    np.save(file_path, G)

# save results_dict
file_name = ('simple_ae_amazon_test_100_'+'emb_%d.npy') % (emb_dim)
file_path = checkpoint_dir + file_name
np.save(file_path, results_dict)
