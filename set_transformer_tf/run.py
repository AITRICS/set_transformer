import os
import time
import argparse
import pickle
import tensorflow as tf

from models import *
from gmm import gmmrnd_easy


parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--enc', type=str, default='ff')
parser.add_argument('--dec', type=str, default='ff')
parser.add_argument('--n_inds', type=int, default=None)
parser.add_argument('--B', type=int, default=10)
parser.add_argument('--N_min', type=int, default=300)
parser.add_argument('--N_max', type=int, default=600)
parser.add_argument('--K', type=int, default=4)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--run_name', type=str, default='trial')
parser.add_argument('--num_steps', type=int, default=50000)
parser.add_argument('--test_freq', type=int, default=200)
parser.add_argument('--save_freq', type=int, default=400)
parser.add_argument('--exp_name', type=str, default='trial')
parser.add_argument('--net', type=str, default='set_transformer')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

# Argument parameters
B = args.B  # batch_size
n_steps = args.num_steps
lr = args.lr
N_min = args.N_min
N_max = args.N_max
K = args.K
D = 2
save_freq = args.save_freq

X = tf.placeholder(tf.float32, [None, None, D])
global_step = tf.train.get_or_create_global_step()
lr = tf.train.piecewise_constant(tf.cast(global_step, dtype=tf.int32), [int(0.7*n_steps)], [lr, 0.1*lr])

# Architecture combinations
enc = args.enc
dec = args.dec
n_inds = args.n_inds
if args.net == 'set_transformer':
    enc = 'sab'
    n_inds = 32
    dec = 'sabsab'
elif args.net == 'deepset':
    enc = 'ff'
    dec = 'ff'
else:
    pass
arch = enc if enc == dec else enc + '_' + dec

# Set directory to save model
if n_inds is not None and enc == 'sab':
    arch += '_' + str(n_inds)
save_dir = os.path.join('./results', arch, args.exp_name)

model = build_model(X, K, D, enc=enc, dec=dec, n_inds=n_inds)

if not os.path.isfile('benchmark.pkl'):
    print('you should generate a benchmark')
else:
    with open('benchmark.pkl', 'rb') as f:
        bench = pickle.load(f)


def train():
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    train_op = tf.train.AdamOptimizer(lr).minimize(-model['ll'][0], global_step=global_step)
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    logfile = open(os.path.join(save_dir, time.strftime('%Y%m%d-%H%M%S') + '.log'), 'wb', 0)

    for t in range(1, n_steps + 1):
        N = np.random.randint(N_min, N_max)
        np_X = gmmrnd_easy(B, N, K, return_meta=False)
        _, np_train_ll = sess.run([train_op, model['ll'][0]], {X: np_X})
        if t % 200 == 0:
            np_lr = sess.run(lr)
            line = 'step {}, lr {:.3e}, train ll {:.4f}'.format(t, np_lr, np_train_ll)
            print("line:", line)
            logfile.write((line + '\n').encode())
            test(sess=sess, logfile=logfile)
        if t % save_freq == 0:
            saver.save(sess, os.path.join(save_dir, 'model'))
    saver.save(sess, os.path.join(save_dir, 'model'))
    logfile.close()



def gen_benchmark():
    N_list = np.random.randint(N_min, N_max, 100)
    data = {}
    data['X'] = []
    data['ll'] = 0.
    for N in N_list:
        batch = gmmrnd_easy(B, N, K)
        data['X'].append(batch['X'])
        data['ll'] += batch['true_ll']
    data['ll'] /= len(N_list)
    with open('benchmark.pkl', 'wb') as f:
        pickle.dump(data, f)


def test(sess=None, logfile=None):
    if sess is None:
        sess = tf.Session()
        tf.train.Saver().restore(sess, os.path.join(save_dir, 'model'))
    if logfile is None:
        logfile = open(os.path.join(save_dir, 'test.log'), 'wb')
    avg_ll = 0
    for np_X in bench['X']:
        avg_ll = np.array(sess.run(model['ll'], {X: np_X}))
    avg_ll /= len(bench['X'])
    line = 'average ll/data {:.4f} -> {:.4f}, '.format(avg_ll[0], avg_ll[1])
    line += '(oracle ll/data {:.4f})'.format(bench['ll'])
    print(line)
    logfile.write((line + '\n\n').encode())


if __name__=='__main__':
    if args.mode == 'train':
        train()
    elif args.mode == 'gen_benchmark':
        gen_benchmark()
    elif args.mode == 'test':
        test()
