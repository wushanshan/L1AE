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

"""Helper functions"""
from __future__ import division
from gurobipy import *
import numpy as np


def prepareSparseTensor(input_csrMat):
    """Extract the indices/values/shapes from a csr_matrix"""
    batch_inputs = input_csrMat.tocoo()
    batch_indices = [batch_inputs.row, batch_inputs.col]
    batch_indices = np.vstack(batch_indices).T.astype(np.int64)
    batch_values = batch_inputs.data.astype(np.float32)
    return batch_indices, batch_values, np.array(input_csrMat.shape)


def LBCS(G, emb_dim, X_train, X_test):
    """
    Learning-based compressive subsampling (LBCS) method
    proposed by Baldassarre et al., 2016.
    """
    num_train = X_train.shape[0]
    Y_train = X_train.dot(G)
    Y_test = X_test.dot(G)
    col_energy = np.dot(np.ones([1, num_train]), Y_train**2)
    sorted_indices = np.argsort(col_energy[0, :])
    A = G[:, sorted_indices[-emb_dim:]]
    Y_test = Y_test[:, sorted_indices[-emb_dim:]]
    return A, Y_test


def l1_min_err(A, y, true_x):
    """
    Solve min_x ||x||_1 s.t. Ax=y, and compute err = ||x-true_x||_2.
    To convert it to the form of an LP:
    min_i \sum_i h_i s.t. -h_i <= x_i <= h_i, Ax=y
    """
    emb_dim, input_dim = A.shape
    model = Model()
    model.params.outputflag = 0   # disable solver output
    x = []
    for i in xrange(input_dim):
        x.append(model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, obj=0))
    for i in xrange(input_dim):
        x.append(model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, obj=1))
    model.update()
    # add inequality constraints
    for i in xrange(input_dim):
        model.addConstr(x[i] - x[i+input_dim] <= 0)
        model.addConstr(x[i] + x[i+input_dim] >= 0)
    # add equality constraints
    for i in xrange(emb_dim):
        coeff = A[i, :]
        expr = LinExpr(coeff, x[:input_dim])
        model.addConstr(lhs=expr, sense=GRB.EQUAL, rhs=y[i])
    # optimize the model and obtain the results
    model.optimize()
    res = []
    for v in model.getVars():
        res.append(v.x)
    return np.linalg.norm(res[:input_dim]-true_x)


def l1_min_pos(A, y, true_x):
    """
    Solve min_x sum_ix_i s.t. Ax=y, x_i>= 0 and compute err = ||x-true_x||_2
    """
    emb_dim, input_dim = A.shape
    model = Model()
    model.params.outputflag = 0  # disable solver output
    x = []
    for i in xrange(input_dim):
        # The lower bound lb=0.0 indicates that x>=0
        x.append(model.addVar(lb=0.0, ub=GRB.INFINITY, obj=1))
    model.update()
    # add equality constraints
    for i in xrange(emb_dim):
        coeff = A[i, :]
        expr = LinExpr(coeff, x)
        model.addConstr(lhs=expr, sense=GRB.EQUAL, rhs=y[i])
    # optimize the model and obtain the results
    model.optimize()
    res = []
    for v in model.getVars():
        res.append(v.x)
    return np.linalg.norm(res[:input_dim]-true_x)


def l1_min_avg_err(A, Y, true_X, use_pos=False, eps=1e-10):
    """
    Run l1_min for each sample, and compute the RMSE.
    true_X is a 2D csr_matrix with shape=(num_sample, input_dim).
    """
    num_sample = Y.shape[0]
    num_exact = 0  # number of samples that are exactly recovered
    num_solved = num_sample  # number of samples that successfully runs l1_min
    err = 0
    for i in xrange(num_sample):
        y = Y[i, :].reshape(-1,)
        x = true_X[i, :].toarray().reshape(-1,)
        try:
            if use_pos:
                temp_err = l1_min_pos(A, y, x)
            else:
                temp_err = l1_min_err(A, y, x)
            if temp_err < eps:
                num_exact += 1
            err += temp_err**2  # squared error
        except Exception:
            num_solved -= 1
    avg_err = np.sqrt(err/num_solved)  # RMSE
    exact_ratio = num_exact/float(num_sample)
    solved_ratio = num_solved/float(num_sample)
    return avg_err, exact_ratio, solved_ratio


def CoSaMP_block_sparsity(A, y, true_x, block_dim, sparsity_level,
                          eps=1e-10, use_pos=False):
    """
    Perform CoSaMP with block-sparsity model.
    Block-sparsity signals are defined in Section VI of
    the paper "Model-based compressive sensing".
    Args:
        A: 2-D array, measurement matrix, shape=(emb_dim, input_dim)
        y: 1-D array, measurement signal, shape=(emb_dim,)
        true_x: 1-D array, the true signal, shape=(input_dim,)
        block_dim: int, length of one block
        sparsity_level: int, number of blocks that the true signal contains,
            therefore, true_x has sparsity_level*block_dim nonzeros.
    Return:
        err = ||x-true_x||_2
    Reference: Algorithm 1 of "Model-based compressive sensing".
    ###############
    while halting criterion false do:
        z = H_model_2[A^t(y-Ax)],
        I = supp(z) join supp(x),
        new_x = arg min_x A_Ix_I=y,
        x = H_model_1(new_x)
    where H_model_a is the hard-thresholding operator that
    keeps the a*sparsity_level*block_dim largest terms.
    ###############
    If use_pos=True, we will use the fact that true_x is non-negative.
    The only change in the algorithm is that before the last step,
    we will set all negative entries of new_x to be zero.
    This change ensures that the output vector is non-negative.
    Besides, it gives the same recovery guarantee as the original algorithm.
    """
    emb_dim, input_dim = A.shape
    x = np.zeros(input_dim)
    residual = y - np.dot(A, x)
    err = np.linalg.norm(x-true_x)
    current_supp_idx = None  # block indices of current estimation's support
    num_block = int(input_dim/block_dim)  # number of total blocks
    while err > eps:
        # Step 1: Form a proxy of the residual.
        #         Identify the location of the largest (l2_norm)
        #         2*sparsity_level blocks.
        proxy = np.dot(np.transpose(A), residual)
        supp_indicator = np.zeros(input_dim, dtype=np.bool)
        for j in xrange(sparsity_level*2):
            block_idx = np.argmax([np.linalg.norm(
                proxy[i*block_dim:(i+1)*block_dim]) for i in range(num_block)])
            supp_indicator[block_idx*block_dim:(block_idx+1)*block_dim] = 1
            # set the entries over the selected block to be zero
            proxy[block_idx*block_dim:(block_idx+1)*block_dim] = 0.0
        # Step 2: Merge the supports of proxy with the current approximation.
        #         Note that current_supp_idx contains sparsity_level integers.
        if current_supp_idx is not None:
            for block_idx in current_supp_idx:
                supp_indicator[block_idx*block_dim:(block_idx+1)*block_dim] = 1
        # Step 3: Solve a least-squares problem over the merged support.
        A_LS = A[:, supp_indicator]
        x_LS = np.dot(np.linalg.pinv(A_LS), y)
        new_x = np.zeros(input_dim)
        new_x[supp_indicator] = x_LS
        # Step 4: Prune the new_x by retaining only the largest (l2_norm)
        #         sparsity_level blocks.
        #         Also update the current approximation.
        #         If use_pos=True, keep only positive entries in new_x.
        if use_pos:
            new_x = np.maximum(new_x, np.zeros(input_dim))
        x = np.zeros(input_dim)
        current_supp_idx = np.zeros(sparsity_level, dtype=int)
        for s in xrange(sparsity_level):
            block_idx = np.argmax([np.linalg.norm(
                new_x[i*block_dim:(i+1)*block_dim]) for i in range(num_block)])
            selected_block = new_x[block_idx*block_dim:(block_idx+1)*block_dim]
            x[block_idx*block_dim:(block_idx+1)*block_dim] = selected_block
            # set the entries over the selected block to be zero
            new_x[block_idx*block_dim:(block_idx+1)*block_dim] = 0.0
            current_supp_idx[s] = block_idx
        # Step 5: Update the the residual and the error.
        residual = y - np.dot(A, x)
        new_err = np.linalg.norm(x-true_x)
        if new_err < err:
            err = new_err
        else:
            break
    return err


def CoSaMP_block_avg_err(A, Y, true_X, block_dim,
                         sparsity_level, eps=1e-10,
                         use_pos=False):
    """
    Run CoSaMP_block_sparsity for each sample, and compute the RMSE.
    true_X is a 2D csr_matrix with shape=(num_sample, input_dim).
    """
    num_sample = Y.shape[0]
    num_exact = 0  # number of samples that are exactly recovered
    num_solved = num_sample  # number of samples that successfully runs CoSaMP
    err = 0
    for i in xrange(num_sample):
        y = Y[i, :].reshape(-1,)
        x = true_X[i, :].toarray().reshape(-1,)
        try:
            temp_err = CoSaMP_block_sparsity(A, y, x,
                                             block_dim=block_dim,
                                             sparsity_level=sparsity_level,
                                             use_pos=use_pos)
            if temp_err < eps:
                num_exact += 1
            err += temp_err**2  # squared error
        except Exception:
            num_solved -= 1
    avg_err = np.sqrt(err/num_solved)  # RMSE
    exact_ratio = num_exact/float(num_sample)
    solved_ratio = num_solved/float(num_sample)
    return avg_err, exact_ratio, solved_ratio


def CoSaMP_onehot_sparsity(A, y, true_x, feature_indices,
                           eps=1e-10, use_pos=False):
    """
    Perform CoSaMP with onehot-sparsity model. Here 'onehot-sparsity' refers to
    the sparsity structure of one-hot encoded data.
    The onehot-sparsity model is specified by 'feature_indices'. It is a 1D
    array storing the index ranges of each categorical feature.
    Refererence: see python doc of sklearn.preprocessing.OneHotEncoder.
    Args:
        A: 2-D array, measurement matrix, shape=(emb_dim, input_dim)
        y: 1-D array, measurement signal, shape=(emb_dim,)
        true_x: 1-D array, the true signal, shape=(input_dim,)
        feature_indices: 1-D array, the onehot-sparsity model,
                         feature_i in the original data is mapped to features
                         from feature_indices[i] to feature_indices[i+1].
    Return:
        err = ||x-true_x||_2
    Reference: Algorithm 1 of "Model-based compressive sensing".
    ###############
    while halting criterion false do:
        z = H_model_2[A^t(y-Ax)],
        I = supp(z) join supp(x),
        new_x = arg min_x A_Ix_I=y,
        x = H_model_1(new_x)
    where H_model_a is the hard-thresholding operator that
    keeps the a*num_feat largest terms, where num_feat=len(feature_indices)-1
    is the number of nonzeros of true_x.
    ###############
    If use_pos=True, we will use the fact that true_x is non-negative.
    The only change in the algorithm is that before the last step,
    we will set all negative entries of new_x to be zero.
    This change ensures that the output vector is non-negative.
    Besides, it gives the same recovery guarantee as the original algorithm.
    """
    emb_dim, input_dim = A.shape
    x = np.zeros(input_dim)
    residual = y - np.dot(A, x)
    err = np.linalg.norm(x-true_x)
    current_supp_idx = None  # feature indices of current estimation's support
    num_feat = len(feature_indices)-1  # number of nonzeros
    while err > eps:
        # Step 1: Form a proxy of the residual.
        #         Identify the location of the largest entry (absolute value)
        #         in each feature range.
        proxy = np.dot(np.transpose(A), residual)
        supp_indicator = np.zeros(input_dim, dtype=np.bool)
        for i in xrange(2):
            for j in xrange(num_feat):
                max_idx = np.argmax(np.abs(
                            proxy[feature_indices[j]:feature_indices[j+1]]))
                supp_indicator[max_idx+feature_indices[j]] = 1
                # set the selected entry to be zero
                proxy[max_idx+feature_indices[j]] = 0.0
        # Step 2: Merge the supports of proxy with the current approximation.
        #         Note that current_supp_idx contains num_feat integers.
        if current_supp_idx is not None:
            for idx in current_supp_idx:
                supp_indicator[idx] = 1
        # Step 3: Solve a least-squares problem over the merged support.
        A_LS = A[:, supp_indicator]
        x_LS = np.dot(np.linalg.pinv(A_LS), y)
        new_x = np.zeros(input_dim)
        new_x[supp_indicator] = x_LS
        # Step 4: Prune the new_x by retaining only the largest
        #         num_feat entries.
        #         Also update the current approximation.
        #         If use_pos=True, keep only positive entries in new_x.
        if use_pos:
            new_x = np.maximum(new_x, np.zeros(input_dim))
        x = np.zeros(input_dim)
        current_supp_idx = np.zeros(num_feat, dtype=int)
        for j in xrange(num_feat):
            max_idx = np.argmax(np.abs(
                        new_x[feature_indices[j]:feature_indices[j+1]]))
            x[max_idx+feature_indices[j]] = new_x[max_idx+feature_indices[j]]
            # set the seletec entry to be zero
            new_x[max_idx+feature_indices[j]] = 0.0
            current_supp_idx[j] = max_idx+feature_indices[j]
        # Step 5: Update the the residual and the error.
        residual = y - np.dot(A, x)
        new_err = np.linalg.norm(x-true_x)
        if new_err < err:
            err = new_err
        else:
            break
    return err


def CoSaMP_onehot_avg_err(A, Y, true_X, feature_indices,
                          eps=1e-10, use_pos=False):
    """
    Run CoSaMP_onehot_sparsity for each sample, and compute the RMSE.
    true_X is a 2D csr_matrix with shape=(num_sample, input_dim).
    """
    num_sample = Y.shape[0]
    num_exact = 0  # number of samples that are exactly recovered
    num_solved = num_sample  # number of samples that successfully runs CoSaMP
    err = 0
    for i in xrange(num_sample):
        y = Y[i, :].reshape(-1,)
        x = true_X[i, :].toarray().reshape(-1,)
        try:
            temp_err = CoSaMP_onehot_sparsity(A, y, x,
                                              feature_indices, eps=eps,
                                              use_pos=use_pos)
            if temp_err < eps:
                num_exact += 1
            err += temp_err**2  # squared error
        except Exception:
            num_solved -= 1
    avg_err = np.sqrt(err/num_solved)  # RMSE
    exact_ratio = num_exact/float(num_sample)
    solved_ratio = num_solved/float(num_sample)
    return avg_err, exact_ratio, solved_ratio
