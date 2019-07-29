import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Activation

softmax = tf.nn.softmax
layer_norm = tf.contrib.layers.layer_norm

# B: number of datasets
# N: number of data points / dataset
# D: dimension
# X: B * N * D
# Y: B * M * D
def MAB(X, Y, n_units, n_heads):
    Q = Dense(n_units)(X)
    K = Dense(n_units)(Y)
    V = Dense(n_units)(Y)

    Q_ = tf.concat(tf.split(Q, n_heads, 2), 0)
    K_ = tf.concat(tf.split(K, n_heads, 2), 0)
    V_ = tf.concat(tf.split(V, n_heads, 2), 0)

    att = softmax(tf.matmul(Q_, tf.transpose(K_, [0, 2, 1])) / tf.sqrt(tf.cast(n_units, tf.float32)))
    outs = tf.matmul(att, V_)
    outs = tf.concat(tf.split(outs, n_heads, 0), 2)
    outs += Q

    outs = layer_norm(outs, begin_norm_axis=-1)
    outs = Activation('relu')(outs + Dense(n_units)(outs))
    outs = layer_norm(outs, begin_norm_axis=-1)

    return outs

def SAB(X, n_units, n_heads):
    return MAB(X, X, n_units, n_heads)

def IMAB(X, n_inds, n_units, n_heads, var_name='ind'):
    I = tf.get_variable(var_name, shape=[1, n_inds, n_units])
    I = tf.tile(I, [tf.shape(X)[0], 1, 1])
    return MAB(I, X, n_units, n_heads)

def ISAB(X, n_inds, n_units, n_heads, var_name='ind'):
    H = IMAB(X, n_inds, n_units, n_heads, var_name=var_name)
    return MAB(X, H, n_units, n_heads)
