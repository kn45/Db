#!/usr/bin/env python
# -*- coding=utf8 -*-

import numpy as np
import sys
sys.path.append('../utils/')
from dataproc import *
from datetime import datetime


expand_feats()

if __name__ == '__main__':
    ifile = open(sys.argv[1])
    ofile = open(sys.argv[2], 'w')
    for ln in ifile:
        flds = ln.rstrip('\n').split(' ')
        label = flds[0:1]
        feats = flds[1:]
        mod_feats = expand_feats(feats)
        print >> ofile, ''.join(label + mod_feats)
    ifile.close()
    ofile.close()
