import numpy as np
import numpy.random as npr
import tensorflow as tf
import tensorflow.contrib.distributions as tfp

MVNDiag = tfp.MultivariateNormalDiag

# M: number of datasets
# N: number of data / dataset
# D: dimension
# K: number of clusters
def gmmrnd_easy(M, N, K, D=2, sigma=0.3, return_meta=True):
    mu = -4 + 8 * npr.random((M, K, D))
    X = np.zeros((M, N, D))

    pi = npr.dirichlet(np.ones(K), M)
    y = np.zeros((M, N), dtype=int)

    for i in range(M):
        y[i] = npr.choice(K, N, p=pi[i])
        X[i] = mu[i][y[i]] + npr.normal(size=[N,D])*sigma

    diff = X[:,:,None,:] - mu[:,None,:,:]
    ll = (-0.5 * (diff ** 2) / (sigma ** 2)).sum(3) - 0.5 * D * np.log(2 * np.pi) - D * np.log(sigma)
    ll += np.log(pi + 1e-10)[:, None, :]
    ll = np.logaddexp(ll, 2)
    ll = ll.mean()

    if return_meta:
        return {'X': X, 'y': y, 'true_ll': ll, 'true_params': {'pi': pi, 'mu': mu, 'sigma': sigma}}
    else:
        return X

# X: M * N * D
# params:
#   pi: M * K
#   mu: M * K * D
#   sigma: M * K * D
def gmm_loglikel(X, params):
    # M * 1 * K * D
    px = MVNDiag(tf.expand_dims(params['mu'], 1), tf.expand_dims(params['sigma'], 1))

    # M * N * K
    ll = tf.log(tf.expand_dims(params['pi'], 1) + 1e-10) + px.log_prob(tf.expand_dims(X, 2))

    # M * N
    y = tf.argmax(ll, 2)

    # M * N * K
    res = tf.exp(ll - tf.reduce_logsumexp(ll, axis=2, keepdims=True))

    # M * N
    ll = tf.reduce_logsumexp(ll, axis=2)

    ll = tf.reduce_mean(ll)

    return ll, y, res


# x: M * N * D
# res: M * N * K
def gmm_em_step(X, res):
    rN = tf.reduce_sum(res, axis=1) + 1e-10

    # M * K
    pi = rN / tf.cast(tf.shape(X)[1], tf.float32)

    # M * K * D
    mu = tf.matmul(tf.transpose(res, [0, 2, 1]), X) / tf.expand_dims(rN, 2)

    # M * N * K * D
    diff = tf.expand_dims(X, 2) - tf.expand_dims(mu, 1)

    # M * K * D
    sigma = tf.reduce_sum(tf.square(diff) * tf.expand_dims(res, 3), 1) / tf.expand_dims(rN, 2)
    sigma = tf.sqrt(sigma + 1e-10)

    return {'pi': pi, 'mu': mu, 'sigma': sigma}