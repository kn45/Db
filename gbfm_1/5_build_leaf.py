#!/usr/bin/env python

import numpy as np
from functools import reduce


leaf_dict_file = 'feat/leaf_dict'

train_leaf_feat = 'feat/trnvld_leaf'
test_leaf_feat = 'feat/test_leaf'
pred_leaf_feat = 'feat/pred_leaf'

train_fm_feat = 'feat/trnvld_leaf.libfm'
test_fm_feat = 'feat/test_leaf.libfm'
pred_fm_feat = 'feat/pred_leaf.libfm'

def build_dict():
    files = ['feat/trnvld_leaf',
        'feat/test_leaf']
    data = []
    for fi in files:
        with open(fi) as f:
            data.extend(map(lambda x: map(int, x.rstrip('\n').split(' ')[1:]), f.readlines()))
    data = np.array(data)
    print data.shape
    with open(leaf_dict_file, 'w') as fo:
        m = data.max(axis=0)
        for l in m:
            print >> fo, l


def trans2libfm(ifn, ofn):
    leaf_dict = [int(x.rstrip('\n')) for x in open(leaf_dict_file).readlines()]
    acc_dict = []
    for i in range(len(leaf_dict)):
        if i > 0:
            acc_dict.append(leaf_dict[i-1] + acc_dict[i-1] + 1)
        else:
            acc_dict.append(0)
    print acc_dict

    with open(ifn) as fi, open(ofn, 'w') as fo:
        for ln in fi:
            flds = ln.rstrip('\n').split(' ')
            label = flds[0]
            ids = map(int, flds[1:])
            t_ids = [acc_dict[i] + x for i, x in enumerate(ids)]
            libfm_ids = [str(x) + ':1' for x in t_ids]
            print >> fo, label + ' ' + ' '.join(libfm_ids)


if __name__ == '__main__':
    #build_dict()
    #trans2libfm(test_leaf_feat, test_fm_feat)
    #trans2libfm(train_leaf_feat, train_fm_feat)
    trans2libfm(pred_leaf_feat, pred_fm_feat)
