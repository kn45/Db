#!/usr/bin/env python

import cPickle
import logging
import numpy as np
import sys
import lightgbm as lgb
from operator import itemgetter

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s]: %(message)s")

# trainf = 'feat/train_feature.libsvm'
trainf = 'feat/trnvld_feature.libsvm'
# trainf = 'feat/all_feature.libsvm'
testf = 'feat/test_feature.libsvm'


def train():
    logging.info('loading training data')
    data_train_dmat = lgb.Dataset(trainf)
    data_test_dmat = lgb.Dataset(testf)

    logging.info('start training')
    params = {
        'application': 'regression',
        'num_leaves': 31,
        'learning_rate': 0.01,
        'min_data_in_leaf': 50,
        'metric': 'mse',
        'min_sum_hessian_in_leaf': 1.,
        'lambda_l1': 1.0,
        'lambda_l2': 1.0,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'max_bin': 63
    }

    cv_params = {
        'params': params,
        'train_set': data_train_dmat,
        'num_boost_round': 3000,  # max round
        'early_stopping_rounds': 200,
        'verbose_eval': True,
        'valid_sets': [data_train_dmat, data_test_dmat]
        #'stratified': False,
        #'nfold': 5
    }
    mdl_bst = lgb.train(**cv_params)
    #mdl_bst = lgb.cv(**cv_params)


if __name__ == '__main__':
    train()
