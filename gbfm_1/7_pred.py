#!/usr/bin/env python


import sys
import tensorflow as tf
sys.path.append('../utils')
import dataproc
from fm import FMRegressor

INP_DIM = 17640
HID_DIM = 4
REG_W = 1.1  # 1st
REG_V = 10.1  # 2nd

LR = 1e-4
TOTAL_ITER = 4000

MDL_CKPT_DIR = './model_ckpt/'
PRED_FILE = 'feat/pred_leaf.libfm'


def inp_fn(data):
    bs = len(data)
    x_idx = []
    x_vals = []
    y_vals = []
    for i, inst in enumerate(data):
        flds = inst.split(' ')
        label = float(flds[0])
        feats = flds[1:]
        for feat in feats:
            idx, val = feat.split(':')
            idx = int(idx)
            val = float(val)
            x_idx.append([i, idx])
            x_vals.append(val)
        y_vals.append([label])
    x_shape = [bs, INP_DIM]
    return (x_idx, x_vals, x_shape), y_vals


with open(PRED_FILE) as fpred:
    pred_data = [x.rstrip('\n') for x in fpred.readlines()]
pred_x, pred_y = inp_fn(pred_data)

mdl = FMRegressor(
    inp_dim=INP_DIM,
    hid_dim=HID_DIM,
    lambda_w=REG_W,
    lambda_v=REG_V,
    lr=LR)

sess = tf.Session()
mdl.saver.restore(sess, tf.train.latest_checkpoint(MDL_CKPT_DIR))
print sess.run(mdl.global_step)
with open('pred/pred_res.csv', 'w') as f:
    preds = mdl.predict(sess, pred_x)
    for p in preds:
        print >> f, p[0]
sess.close()
