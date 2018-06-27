"""Define the autoencoder"""
from __future__ import division
from time import time
from utils import prepareSparseTensor

import numpy as np
import tensorflow as tf


class L1AE(object):
    def __init__(self, sess, input_dim, emb_dim, decoder_num_steps):
        self.sess = sess
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.decoder_num_steps = decoder_num_steps
        # define the input as a SparseTensor
        self.indices_x = tf.placeholder("int64", [None, 2])
        self.values_x = tf.placeholder("float", [None])
        self.dense_shape_x = tf.placeholder("int64", [2])
        self.input_x = tf.SparseTensor(indices=self.indices_x,
                                       values=self.values_x,
                                       dense_shape=self.dense_shape_x)
        self.encoder_weight = tf.Variable(tf.truncated_normal(
                                    [self.input_dim, self.emb_dim],
                                    stddev=1.0/np.sqrt(self.input_dim)))
        # encode the input
        self.encode = tf.sparse_tensor_dense_matmul(self.input_x,
                                                    self.encoder_weight)
        # decode by simulating decoder_num_steps projected subgradient updates
        self.step_size = tf.Variable(1.0)

        def decode_subgrad(x, W, num_steps, step_size):
            """
            Simulates several steps of subgradient descent of an l1-min:
            x+ = x + step_size*(W^TW-I)sign(x)
            """
            x = tf.matmul(x, W, transpose_b=True)
            for i in xrange(num_steps):
                x = x + (tf.matmul(tf.matmul(tf.sign(x), W), W,
                         transpose_b=True)-tf.sign(x))*(step_size/(i+1))
                x = tf.layers.batch_normalization(x, axis=1)
            return tf.nn.relu(x)
        self.pred = decode_subgrad(self.encode, self.encoder_weight,
                                   self.decoder_num_steps, self.step_size)
        # define the squared loss
        self.sq_loss = tf.reduce_mean(tf.pow(tf.sparse_add(self.input_x,
                                      -self.pred), 2))*self.input_dim
        self.learning_rate = tf.placeholder("float", [])
        self.sq_optim = tf.train.GradientDescentOptimizer(
                                     self.learning_rate).minimize(self.sq_loss)

    def train(self, X_train, X_valid, batch_size, learning_rate,
              max_training_epochs=2e4, display_interval=1e2,
              validation_interval=10, max_steps_not_improve=5):
        """Perform training on the model
        Args:
            max_training_epochs [2e4]: stop training after 2e4 epochs.
            display_interval [100]: print the training info every 100 epochs.
            validation_interval [10]: compute validation loss every 10 epochs.
            max_steps_not_improve [5]: stop training when the validation loss
                                does not improve for 5 validation_intervals.
        """
        # initialize the variables
        init = tf.global_variables_initializer()
        self.sess.run(init)
        # early-stopping parameters
        best_valid_loss = self.inference(X_valid, batch_size)
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
                _, c = self.sess.run([self.sq_optim, self.sq_loss], feed_dict={
                                      self.indices_x: indices,
                                      self.values_x: values,
                                      self.dense_shape_x: shape,
                                      self.learning_rate: learning_rate})
                train_loss += c
            if current_epoch % validation_interval == 0:
                current_valid_loss = self.inference(X_valid, batch_size)
                if current_valid_loss < best_valid_loss:
                    best_valid_loss = current_valid_loss
                    num_steps_not_improve = 0
                else:
                    num_steps_not_improve += 1
            if current_epoch % display_interval == 0:
                # print avg_err,
                print("Epoch: %05d" % (current_epoch),
                      "TrainSqErr: %f" % (train_loss/total_batch),
                      "ValidSqErr: %f" % (current_valid_loss),
                      "StepSize: %f" % (self.sess.run(self.step_size)))
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

    def inference(self, X, batch_size):
        """Perform inference on the model"""
        batch_size = np.amin([batch_size, X.shape[0]])
        total_batch = int(X.shape[0]/batch_size)
        total_loss = 0
        # loop over all batches
        for batch_i in xrange(total_batch):
            inputs = X[batch_i*batch_size:(batch_i+1)*batch_size, :]
            indices, values, shape = prepareSparseTensor(inputs)
            # get the loss value
            c = self.sess.run(self.sq_loss, feed_dict={
                                            self.indices_x: indices,
                                            self.values_x: values,
                                            self.dense_shape_x: shape})
            total_loss += c
        return total_loss/total_batch
