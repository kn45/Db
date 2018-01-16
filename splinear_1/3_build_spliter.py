#!/usr/bin/env python

import sys
sys.path.append('../utils')
from dataproc import *

spliter = BinSpliter()
data = {}

data_file = sys.argv[1]
with open(data_file) as fin:
    for ln in fin:
        rec = ln.rstrip('\n').split(' ')[1:]
        for feat in rec:
            idx, val = feat.split(':')
            if idx not in data:
                data[idx] = []
            data[idx].append(float(val))

for idx in data:
    spliter.add_bin(data[idx], 'f'+idx, 32)
spliter.save_bin('feat_bin')
