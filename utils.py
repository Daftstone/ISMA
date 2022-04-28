from time import time
from time import strftime
from time import localtime
import os
import tensorflow as tf
import numpy as np
import math
import copy
import scipy.sparse as sp

flags = tf.flags
FLAGS = flags.FLAGS

def uniform(shape, scale=0.05, name=None):
    """Uniform init."""
    initial = tf.random_uniform(shape, minval=-scale, maxval=scale, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def zeros(shape, name=None):
    """All zeros."""
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def ones(shape, name=None):
    """All ones."""
    initial = tf.ones(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)

def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)


def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)

def transform_pairwise(X, y):
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    idx_pos = np.random.choice(idx_pos, len(X))
    idx_neg = np.random.choice(idx_neg, len(X))
    return X[idx_pos], X[idx_neg], np.ones_like(y)

def transform_pairwise_weibo(X, y):
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    idx_pos = np.random.choice(idx_pos, len(X))
    idx_neg = np.random.choice(idx_neg, len(X))
    return X[idx_pos], X[idx_neg], np.ones_like(y)


def pert_vector_product(ys, xs1, xs2, v, do_not_sum_up=True):
    # Validate the input
    length = len(xs1)
    if len(v) != length:
        raise ValueError("xs and v must have the same length.")
    # First backprop
    grads = tf.gradients(ys, xs1)

    # grads = xs
    assert len(grads) == length
    elemwise_products = [
        tf.multiply(grad_elem, tf.stop_gradient(v_elem))
        for grad_elem, v_elem in zip(grads, v) if grad_elem is not None
    ]
    # Second backprop
    if do_not_sum_up:
        seperate = []
        for i in range(length):
            seperate.append(tf.gradients(elemwise_products[i], xs2)[0])
        grads_with_none = seperate
    else:
        grads_with_none = tf.gradients(elemwise_products, xs2)

    return_grads = [grad_elem if grad_elem is not None \
                        else tf.zeros_like(xs2) \
                    for grad_elem in grads_with_none]
    return return_grads


def hessian_vector_product(ys, xs, v, do_not_sum_up=True, scales=1.):
    # Validate the input
    length = len(xs)
    if len(v) != length:
        raise ValueError("xs and v must have the same length.")
    # First backprop
    grads = tf.gradients(ys, xs)

    # grads = xs
    assert len(grads) == length
    elemwise_products = [
        tf.multiply(grad_elem, tf.stop_gradient(v_elem))
        for grad_elem, v_elem in zip(grads, v) if grad_elem is not None
    ]
    # Second backprop
    if do_not_sum_up:
        seperate = []
        for i in range(length):
            seperate.append(tf.gradients(elemwise_products[i] / scales, xs[i])[0])
        grads_with_none = seperate
    else:
        grads_with_none = tf.gradients(elemwise_products / scales, xs)

    return_grads = [grad_elem if grad_elem is not None \
                        else tf.zeros_like(x) \
                    for x, grad_elem in zip(xs, grads_with_none)]
    return return_grads
