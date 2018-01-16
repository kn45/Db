#!/usr/bin/env python
# -*- coding=utf8 -*-

import numpy as np
import sys
sys.path.append('../utils/')
from dataproc import *
from datetime import datetime


gender_encoder = DictTable({u'男': 0, u'女': 1})


def day_diff(d1, d2):
    d1 = datetime.strptime(d1, "%d/%m/%Y")
    d2 = datetime.strptime(d2, "%d/%m/%Y")
    return abs((d2 - d1).days)


def gender_transform(gender_cf):
    idx = gender_encoder.lookup(gender_cf)
    if idx[0] is None:
        return ['']
    else:
        return idx


def date_transform(dt):
    res = []
    for d in dt:
        res.append(day_diff(d, '01/09/2017'))
    return res


def drop_cf(cf, st, ed):
    return cf[:st] + cf[ed:]


def extend_cf(cf, st, ed, foo):
    return cf[:st] + foo(cf[st:ed]) + cf[ed:]


def cf2libsvm(label, cf):
    fstr = ''
    for idx, val in enumerate(cf):
        if val != '':
            fstr += str(idx+1) + ':' + str(val) + ' '
    if label == '':
        label = '0'
    return label + ' ' + fstr.rstrip(' ')


if __name__ == '__main__':
    ifile = open(sys.argv[1])
    ofile = open(sys.argv[2], 'w')
    for ln in ifile:
        flds = ln.decode('utf8').rstrip('\n').split('\t')
        label = flds[0]
        feats = flds[1:]
        # feats = drop_cf(feats, 3, 4)
        feats = extend_cf(feats, 3, 4, date_transform)
        feats = extend_cf(feats, 1, 2, gender_transform)
        print >> ofile, cf2libsvm(label, feats)
    ifile.close()
    ofile.close()
