import tensorflow as tf
import numpy as np

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.activations import softmax
from tensorflow.python.keras.layers import Dense, Activation
from attention import *
from gmm import *


def perm_equiv(X, n_hid, activation=None, ops='mean'):
    ops = tf.reduce_mean if ops == 'mean' else tf.reduce_max
    Xm = ops(X, axis=1, keepdims=True)
    X = Dense(n_hid)(X) - Dense(n_hid)(Xm)
    X = X if activation is None else Activation('relu')(X)
    return X


def encode(X, arch='ff', n_inds=None):
    if arch == 'ff':
        X = Activation('relu')(Dense(128)(X))
        X = Activation('relu')(Dense(128)(X))
        X = Activation('relu')(Dense(128)(X))
        X = Dense(128)(X)
        # model = Sequential([
        #     Dense(128),
        #     Activation('relu'),
        #     Dense(128),
        #     Activation('relu'),
        #     Dense(128),
        #     Activation('relu'),
        #     Dense(128)
        # ])
        # X = model(X)
    elif arch == 'perm_eq_mean':
        X = perm_equiv(X, 128, activation='relu')
        X = perm_equiv(X, 128, activation='relu')
        X = perm_equiv(X, 128, activation='relu')
        X = perm_equiv(X, 128)
    elif arch == 'perm_eq_max':
        X = perm_equiv(X, 128, activation='relu', ops='max')
        X = perm_equiv(X, 128, activation='relu', ops='max')
        X = perm_equiv(X, 128, activation='relu', ops='max')
        X = perm_equiv(X, 128, ops='max')
    elif arch == 'sab':
        if n_inds is None:
            X = SAB(X, 128, 4)
            X = SAB(X, 128, 4)
        else:
            X = ISAB(X, n_inds, 128, 4, var_name='ind1')
            X = ISAB(X, n_inds, 128, 4, var_name='ind2')
    else:
        raise ValueError('Invalid encoder architecture')
    return X


def decode(X, shape, arch='ff'):
    if arch == 'ff':
        X = tf.reduce_mean(X, axis=1)

        X = Activation('relu')(Dense(128)(X))
        X = Activation('relu')(Dense(128)(X))
        X = Activation('relu')(Dense(128)(X))
        X = Dense(np.prod(shape))(X)

        # model = Sequential([
        #     Dense(128),
        #     Activation('relu'),
        #     Dense(128),
        #     Activation('relu'),
        #     Dense(128),
        #     Activation('relu'),
        #     Dense(np.prod(shape))
        # ])
        # X = model(X)

        X = tf.reshape(X, [-1] + shape)
    elif arch == 'sab':
        K = shape[0]
        X = IMAB(X, K, 128, 4, var_name='seed')
        X = SAB(X, 128, 4)
        X = Dense(np.prod(shape[1:]))(X)
        X = tf.reshape(X, [-1] + shape)
    elif arch == 'sabsab':
        K = shape[0]
        X = IMAB(X, K, 128, 4, var_name='seed')
        X = SAB(X, 128, 4)
        X = SAB(X, 128, 4)
        X = Dense(np.prod(shape[1:]))(X)
        X = tf.reshape(X, [-1] + shape)
    elif arch == 'dotprod':
        C = Activation('tanh')(Dense(128)(X))
        S = softmax(C, axis=1)
        X = tf.reduce_sum(X * S, axis=1)

        X = Activation('relu')(Dense(128)(X))
        X = Activation('relu')(Dense(128)(X))
        X = Activation('relu')(Dense(128)(X))
        X = Dense(np.prod(shape))(X)

        # model = Sequential([
        #     Dense(128),
        #     Activation('relu'),
        #     Dense(128),
        #     Activation('relu'),
        #     Dense(128),
        #     Activation('relu'),
        #     Dense(np.prod(shape))
        # ])
        # X = model(X)

        X = tf.reshape(X, [-1] + shape)
    else:
        raise ValueError('Invalid decoder architecture')
    return X


def build_model(X, K, D, enc='ff', dec='ff', n_inds=None, n_unrolls=1):
    with tf.compat.v1.variable_scope('encoder', reuse=tf.compat.v1.AUTO_REUSE):
        outs = encode(X, arch=enc, n_inds=n_inds)
    with tf.compat.v1.variable_scope('decoder', reuse=tf.compat.v1.AUTO_REUSE):
        outs = decode(outs, [K, 1+2*D], arch=dec)

        # TODO:: explanation of pi, mu, and sigma
        pi_uc = outs[:, :, 0]
        pi = tf.nn.softmax(pi_uc)

        mu = outs[:, :, 1: D+1]

        sigma_uc = outs[:, :, D+1:]
        sigma = tf.nn.softplus(sigma_uc)

    model = dict()
    model['params'] = [{'pi': pi, 'mu': mu, 'sigma': sigma}]
    ll, y, res = gmm_loglikel(X, model['params'][0])
    model['ll'] = [ll]
    model['y'] = [y]
    for i in range(n_unrolls):
        model['params'].append(gmm_em_step(X, res))
        ll, y, res = gmm_loglikel(X, model['params'][-1])
        model['ll'].append(ll)
        model['y'].append(y)
    return model


