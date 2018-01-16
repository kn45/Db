#!/usr/bin/env python
# -*- coding=utf8 -*-

import numpy as np
import sys
sys.path.append('../utils/')
from dataproc import *
from datetime import datetime


bs = BinSpliter()
bs.load_bin('feat_bin')

def expand_feats(feats):
    res = []
    for feat in feats:
        idx, val = feat.split(':')
        new_idx = bs.find_bin('f'+idx, float(val))
        new_idx = (int(idx) - 1) * 32 + new_idx + 1
        res.append(str(new_idx) + ':1')
    return res

if __name__ == '__main__':
    ifile = open(sys.argv[1])
    ofile = open(sys.argv[2], 'w')
    for ln in ifile:
        flds = ln.rstrip('\n').split(' ')
        label = flds[0:1]
        feats = flds[1:]
        mod_feats = expand_feats(feats)
        print >> ofile, ' '.join(label + mod_feats)
    ifile.close()
    ofile.close()
