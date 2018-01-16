#!/usr/bin/env python

import cPickle
import logging
import numpy as np
import sys
import xgboost as xgb
from operator import itemgetter

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s]: %(message)s")

# trainf = 'feat/train_feature.libsvm'
# trainf = 'feat/trnvld_feature.libsvm'
trainf = 'feat/all_feature.libsvm'


def train():
    logging.info('loading training data')
    data_train_dmat = xgb.DMatrix(trainf)

    logging.info('start training')
    bst_params = {
        'nthread': 4,
        'silent': 1,
        'eta': 0.01,
        'gamma': 1.0,
        'eval_metric': ['rmse'],
        'max_depth': 5,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'objective': 'reg:linear',
        'min_child_weight': 100,
        'lambda': 1.0}
    cv_params = {
        'params': bst_params,
        'dtrain': data_train_dmat,
        'num_boost_round': 3000,  # max round
        'nfold': 4,
        'metrics': 'rmse',
        'maximize': False,
        'early_stopping_rounds': 200,
        'verbose_eval': True}
    mdl_bst = xgb.cv(**cv_params)


if __name__ == '__main__':
    train()
