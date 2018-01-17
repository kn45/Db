#!/usr/bin/env python

import cPickle
import logging
import math
import numpy as np
import sys
import xgboost as xgb
from operator import itemgetter

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s]: %(message)s")
model_file = 'gbt_model.pkl'


def pred():
    predf = 'feat/pred_feature.libsvm'
    resf = open('pred/pred_res.csv', 'w')
    data_pred_dmat = xgb.DMatrix(predf)

    # init gbt
    mdl_bst = cPickle.load(open(model_file, 'rb'))
    print 'best iteration:', mdl_bst.best_iteration
    mdl_bst.set_param('nthread', 1)
    pred_res = mdl_bst.predict(
        data_pred_dmat,
        ntree_limit=mdl_bst.best_iteration)
    for p in pred_res:
        print >> resf, math.exp(p)
    resf.close()


if __name__ == '__main__':
    pred()
