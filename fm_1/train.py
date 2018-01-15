#!/usr/bin/env python

import numpy as np
import sys
sys.path.append('../utils')
import tensorflow as tf
from fm import FMRegressor
import dataproc


INP_DIM =
HID_DIM = 32
REG_W = 0.0
REG_V = 0.0


def inp_fn_unilabel(data):
    inp_x = []
    inp_y = []
    for inst in data:
        flds = inst.split('\t')
        label = map(int, flds[0:1])
        feats = map(int, flds[1:])
        inp_y.append(label)
        inp_x.append(dataproc.zero_padding(feats, SEQ_LEN))
    return np.array(inp_x), np.array(inp_y)

train_file = './rt-polarity.shuf.train'
test_file = './rt-polarity.shuf.test'
freader = dataproc.BatchReader(train_file)
with open(test_file) as f:
    test_data = [x.rstrip('\n') for x in f.readlines()]
test_x, test_y = inp_fn_unilabel(test_data)

mdl = FMRegressor(
    inp_dim=INP_DIM,
    hid_dim=HID_DIM,
    lambda_w=,
    lambda_v=,
    lr=1e-4)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())
metrics = ['loss', 'accuracy', 'auc']
niter = 0
mdl_ckpt_dir = './model_ckpt/model.ckpt'
while niter < 1000:
    niter += 1
    batch_data = freader.get_batch(128)
    if not batch_data:
        break
    train_x, train_y = inp_fn_unilabel(batch_data)
    mdl.train_step(sess, train_x, train_y)
    train_eval = mdl.eval_step(sess, train_x, train_y, metrics)
    test_eval = mdl.eval_step(sess, test_x, test_y, metrics) \
        if niter % 50 == 0 else 'SKIP'
    print niter, 'train:', train_eval, 'test:', test_eval
save_path = mdl.saver.save(sess, mdl_ckpt_dir, global_step=mdl.global_step)
print "model saved:", save_path

with open('train_done_test_res', 'w') as f:
    preds = mdl.predict_proba(sess, test_x)
    for l, p in zip(test_y, preds):
        print >> f, '\t'.join(map(str, [l[0], p[1]]))

sess.close()
