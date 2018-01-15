#!/usr/bin/env python

import cPickle
import logging
import numpy as np
import sys
import xgboost as xgb
from operator import itemgetter

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s]: %(message)s")

model_file = 'gbt_model.pkl'
# trainf = 'feat/train_feature.libsvm'
# validf = 'feat/valid_feature.libsvm'
trainf = 'feat/trnvld_feature.libsvm'
validf = 'feat/test_feature.libsvm'


def train():
    logging.info('loading training data')
    data_train_dmat = xgb.DMatrix(trainf)
    data_valid_dmat = xgb.DMatrix(validf)

    logging.info('start training')
    bst_params = {
        'nthread': 4,
        'silent': 1,
        'eta': 0.01,
        'eval_metric': ['rmse'],
        'max_depth': 4,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'objective': 'reg:linear',
        'lambda': 10.0}
    train_params = {
        'params': bst_params,
        'dtrain': data_train_dmat,
        'num_boost_round': 3000,  # max round
        'evals': [(data_train_dmat, 'train'), (data_valid_dmat, 'valid_0')],
        'maximize': False,
        'early_stopping_rounds': 100,
        'verbose_eval': True}
    mdl_bst = xgb.train(**train_params)

    logging.info('Saving model')
    # not use save_model mothod because it cannot dump best_iteration etc.
    cPickle.dump(mdl_bst, open(model_file, 'wb'))

    feat_imp = mdl_bst.get_score(importance_type='gain').items()
    print sorted(feat_imp, key=itemgetter(1), reverse=True)[0:10]


def test():
    testf = 'feat/test_feature.libsvm'
    resf = open('test_res', 'w')
    data_test_dmat = xgb.DMatrix(testf)

    # init gbt
    mdl_bst = cPickle.load(open(model_file, 'rb'))
    mdl_bst.set_param('nthread', 1)
    mdl_bst.set_param('eval_metric', 'rmse')

    test_metric = mdl_bst.eval_set([(data_test_dmat, 'test_0')])
    print float(test_metric.split(':')[-1]) ** 2 / 2.
    pred_res = mdl_bst.predict(
        data_test_dmat,
        ntree_limit=mdl_bst.best_iteration)
    for i in pred_res:
        print >> resf, i
    resf.close()


if __name__ == '__main__':
    train()
    test()
