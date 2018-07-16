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
"""
Define the baseline algorithms. They differ by 1) the measurement matrix,
and 2) the recovery algorithm:
    Gaussian + model-based CoSaMP
    Fourier + model-based CoSaMP
    Gaussian + l1
    Fourier + l1
    PCA + l1
    PCA
    Simple AE + l1
For each baseline algorithm, we compute the RMSE (root mean squared error),
and the percentage of samples that can be exactly recovered.
We also evaluate the performance after adding the *nonnegative* constraint
to the decoding algorithms (both l1 and CoSaMP).
"""
from __future__ import division
from utils import l1_min_avg_err, CoSaMP_block_avg_err, CoSaMP_onehot_avg_err
from sklearn.decomposition import TruncatedSVD
from utils import prepareSparseTensor
from time import time

import numpy as np
import tensorflow as tf


def block_sparse_CoSaMP(X, input_dim, emb_dim, block_dim, sparsity_level):
    """
    Perform model-based CoSaMP on a dataset with block-sparsity.
    Args:
        X: csr_matrix, shape=(num_sample, input_dim)
    """
    # random Gaussian matrix
    G = np.random.randn(input_dim, emb_dim)/np.sqrt(emb_dim)
    Y = X.dot(G)  # sparse.csr_matrix.dot
    g_err, g_exact, _ = CoSaMP_block_avg_err(np.transpose(G), Y, X,
                                             block_dim, sparsity_level,
                                             use_pos=False)
    g_err_pos, g_exact_pos, _ = CoSaMP_block_avg_err(np.transpose(G), Y, X,
                                                     block_dim, sparsity_level,
                                                     use_pos=True)
    # random discrete Fourier transform matrix
    F = np.zeros((input_dim, emb_dim))
    # select emb_dim/2 rows since the DFT matrix is complex
    for col in xrange(int(emb_dim/2)):
        k = np.random.choice(input_dim, 1)
        for row in xrange(input_dim):
            F[row, 2*col] = np.cos(-(row*k*2*np.pi)/input_dim)
            F[row, 2*col+1] = np.sin(-(row*k*2*np.pi)/input_dim)
    F = F/np.sqrt(emb_dim/2)
    Y = X.dot(F)  # sparse.csr_matrix.dot
    f_err, f_exact, _ = CoSaMP_block_avg_err(np.transpose(F), Y, X,
                                             block_dim, sparsity_level,
                                             use_pos=False)
    f_err_pos, f_exact_pos, _ = CoSaMP_block_avg_err(np.transpose(F), Y, X,
                                                     block_dim, sparsity_level,
                                                     use_pos=True)
    res = {}
    res['cosamp_g_err'] = g_err
    res['cosamp_g_exact'] = g_exact
    res['cosamp_g_err_pos'] = g_err_pos
    res['cosamp_g_exact_pos'] = g_exact_pos
    res['cosamp_f_err'] = f_err
    res['cosamp_f_exact'] = f_exact
    res['cosamp_f_err_pos'] = f_err_pos
    res['cosamp_f_exact_pos'] = f_exact_pos
    return res


def onehot_sparse_CoSaMP(X, input_dim, emb_dim, feature_indices):
    """
    Perform model-based CoSaMP on a dataset with onehot-sparsity.
    Args:
        X: csr_matrix, shape=(num_sample, input_dim)
    """
    # random Gaussian matrix
    G = np.random.randn(input_dim, emb_dim)/np.sqrt(emb_dim)
    Y = X.dot(G)  # sparse.csr_matrix.dot
    g_err, g_exact, _ = CoSaMP_onehot_avg_err(np.transpose(G), Y, X,
                                              feature_indices,
                                              use_pos=False)
    g_err_pos, g_exact_pos, _ = CoSaMP_onehot_avg_err(np.transpose(G), Y, X,
                                                      feature_indices,
                                                      use_pos=True)
    # random discrete Fourier transform matrix
    F = np.zeros((input_dim, emb_dim))
    # select emb_dim/2 rows since the DFT matrix is complex
    for col in xrange(int(emb_dim/2)):
        k = np.random.choice(input_dim, 1)
        for row in xrange(input_dim):
            F[row, 2*col] = np.cos(-(row*k*2*np.pi)/input_dim)
            F[row, 2*col+1] = np.sin(-(row*k*2*np.pi)/input_dim)
    F = F/np.sqrt(emb_dim/2)
    Y = X.dot(F)  # sparse.csr_matrix.dot
    f_err, f_exact, _ = CoSaMP_onehot_avg_err(np.transpose(F), Y, X,
                                              feature_indices,
                                              use_pos=False)
    f_err_pos, f_exact_pos, _ = CoSaMP_onehot_avg_err(np.transpose(F), Y, X,
                                                      feature_indices,
                                                      use_pos=True)
    res = {}
    res['cosamp_g_err'] = g_err
    res['cosamp_g_exact'] = g_exact
    res['cosamp_g_err_pos'] = g_err_pos
    res['cosamp_g_exact_pos'] = g_exact_pos
    res['cosamp_f_err'] = f_err
    res['cosamp_f_exact'] = f_exact
    res['cosamp_f_err_pos'] = f_err_pos
    res['cosamp_f_exact_pos'] = f_exact_pos
    return res


def l1_min(X, input_dim, emb_dim):
    """
    Args:
        X: csr_matrix, shape=(num_sample, input_dim)
    """
    # random Gaussian matrix
    G = np.random.randn(input_dim, emb_dim)/np.sqrt(emb_dim)
    Y = X.dot(G)  # sparse.csr_matrix.dot
    g_err, g_exact, _ = l1_min_avg_err(np.transpose(G), Y, X, use_pos=False)
    g_err_pos, g_exact_pos, _ = l1_min_avg_err(np.transpose(G), Y, X,
                                               use_pos=True)
    # random discrete Fourier transform matrix
    F = np.zeros((input_dim, emb_dim))
    # select emb_dim/2 rows since the DFT matrix is complex
    for col in xrange(int(emb_dim/2)):
        k = np.random.choice(input_dim, 1)
        for row in xrange(input_dim):
            F[row, 2*col] = np.cos(-(row*k*2*np.pi)/input_dim)
            F[row, 2*col+1] = np.sin(-(row*k*2*np.pi)/input_dim)
    F = F/np.sqrt(emb_dim/2)
    Y = X.dot(F)  # sparse.csr_matrix.dot
    f_err, f_exact, _ = l1_min_avg_err(np.transpose(F), Y, X, use_pos=False)
    f_err_pos, f_exact_pos, _ = l1_min_avg_err(np.transpose(F), Y, X,
                                               use_pos=True)
    res = {}
    res['l1_g_err'] = g_err
    res['l1_g_exact'] = g_exact
    res['l1_g_err_pos'] = g_err_pos
    res['l1_g_exact_pos'] = g_exact_pos
    res['l1_f_err'] = f_err
    res['l1_f_exact'] = f_exact
    res['l1_f_err_pos'] = f_err_pos
    res['l1_f_exact_pos'] = f_exact_pos
    return res


def PCA_l1(X_train, X_test, input_dim, emb_dim):
    """
    Args:
        X_train: csr_matrix, shape=(num_train_sample, input_dim)
        X_test: csr_matrix, shape=(num_test_sample, input_dim)
    """
    svd = TruncatedSVD(n_components=emb_dim)
    svd.fit(X_train)
    G = svd.components_
    Y = svd.transform(X_test)
    p_err, p_exact, _ = l1_min_avg_err(G, Y, X_test, use_pos=False)
    p_err_pos, p_exact_pos, _ = l1_min_avg_err(G, Y, X_test,
                                               use_pos=True)
    pca_err = 0  # sqaured err after inverse transform
    batch_size = np.amin([256, X_test.shape[0]])  # compute error in batches
    num_batches = int(X_test.shape[0]/batch_size)
    for batch_i in xrange(num_batches):
        start_idx = batch_i*batch_size
        end_idx = (batch_i+1)*batch_size
        X_hat = svd.inverse_transform(Y[start_idx:end_idx, :])  # dense array
        X_true = X_test[start_idx:end_idx, :].toarray()
        pca_err += np.linalg.norm(X_true-X_hat)**2  # squared error
    pca_err = np.sqrt(pca_err/(num_batches*batch_size))  # RMSE
    res = {}
    res['l1_p_err'] = p_err
    res['l1_p_exact'] = p_exact
    res['l1_p_err_pos'] = p_err_pos
    res['l1_p_exact_pos'] = p_exact_pos
    res['pca_err'] = pca_err
    return res


def simple_AE_l1(input_dim, emb_dim, X_train, X_valid, X_test, batch_size,
                 learning_rate_value, max_training_epochs, display_interval,
                 validation_interval, max_steps_not_improve):
    """
    Train a simple autoencoder (with one-layer NN as the decoder).
    """
    # build the graph
    indices_x = tf.placeholder("int64", [None, 2])
    values_x = tf.placeholder("float", [None])
    dense_shape_x = tf.placeholder("int64", [2])
    input_x = tf.SparseTensor(indices=indices_x,
                              values=values_x,
                              dense_shape=dense_shape_x)
    encoder_weight = tf.Variable(tf.truncated_normal(
                                [input_dim, emb_dim],
                                stddev=1.0/np.sqrt(input_dim)))
    decoder_weight = tf.Variable(tf.truncated_normal(
                                [emb_dim, input_dim],
                                stddev=1.0/np.sqrt(input_dim)))
    encode = tf.sparse_tensor_dense_matmul(input_x, encoder_weight)
    decode = tf.nn.relu(tf.matmul(encode, decoder_weight))
    sq_loss = tf.reduce_mean(tf.pow(tf.sparse_add(input_x,
                                    - decode), 2))*input_dim
    learning_rate = tf.placeholder("float", [])
    sq_optim = tf.train.GradientDescentOptimizer(
                                 learning_rate).minimize(sq_loss)

    # define the inference function
    def AE_inference(sess, X, batch_size):
        """Perform inference on the autoencoder"""
        batch_size = np.amin([batch_size, X.shape[0]])
        total_batch = int(X.shape[0]/batch_size)
        total_loss = 0
        # loop over all batches
        for batch_i in xrange(total_batch):
            inputs = X[batch_i*batch_size:(batch_i+1)*batch_size, :]
            indices, values, shape = prepareSparseTensor(inputs)
            # get the loss value
            c = sess.run(sq_loss, feed_dict={indices_x: indices,
                                             values_x: values,
                                             dense_shape_x: shape})
            total_loss += c
        return total_loss/total_batch

    # start the session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    sess.run(init)
    # early stopping parameters
    best_valid_loss = AE_inference(sess, X_valid, batch_size)
    num_steps_not_improve = 0

    # start training
    t0 = time()
    batch_size = np.amin([batch_size, X_train.shape[0]])
    total_batch = int(X_train.shape[0]/batch_size)
    # training cycle
    current_epoch = 0
    while current_epoch < max_training_epochs:
        train_loss = 0
        # random shuffle
        idx = np.random.permutation(X_train.shape[0])
        # Loop over all batches
        for batch_i in xrange(total_batch):
            idx_batch_i = idx[batch_i*batch_size:(batch_i+1)*batch_size]
            train = X_train[idx_batch_i, :]
            indices, values, shape = prepareSparseTensor(train)
            # optimize the sq_loss
            _, c = sess.run([sq_optim, sq_loss], feed_dict={
                                  indices_x: indices,
                                  values_x: values,
                                  dense_shape_x: shape,
                                  learning_rate: learning_rate_value})
            train_loss += c
        # early stopping
        if current_epoch % validation_interval == 0:
            current_valid_loss = AE_inference(sess, X_valid, batch_size)
            if current_valid_loss < best_valid_loss - 5e-4:
                best_valid_loss = current_valid_loss
                num_steps_not_improve = 0
            else:
                num_steps_not_improve += 1
        if current_epoch % display_interval == 0:
            # print avg_err,
            print("Epoch: %05d" % (current_epoch),
                  "TrainSqErr: %f" % (train_loss/total_batch),
                  "ValidSqErr: %f" % (current_valid_loss))
        current_epoch += 1
        # stop training when the validation loss
        # does not improve for certain number of steps
        if num_steps_not_improve > max_steps_not_improve:
            break
    print("Optimization Finished!")
    t1 = time()
    print("Training takes %d epochs in %f secs" % (current_epoch, t1-t0))
    print("Training loss: %f" % (train_loss/total_batch))
    print("Validation loss: %f" % (current_valid_loss))
    test_sq_loss = AE_inference(sess, X_test, batch_size)
    G = sess.run(encoder_weight)
    sess.close()
    return test_sq_loss, G
