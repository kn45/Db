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
trainf = 'feat/trnvld_feature.libsvm'
testf = 'feat/test_feature.libsvm'
predf = 'feat/pred_feature.libsvm'

train_leaf_feat = 'feat/trnvld_leaf'
test_leaf_feat = 'feat/test_leaf'
pred_leaf_feat = 'feat/pred_leaf'

def train():
    logging.info('loading training data')
    data_train_dmat = xgb.DMatrix(trainf)
    # data_valid_dmat = xgb.DMatrix(validf)

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
        'alpha': 1.0,
        'lambda': 1.0}
    train_params = {
        'params': bst_params,
        'dtrain': data_train_dmat,
        'num_boost_round': 837,  # max round
        # 'evals': [(data_train_dmat, 'train'), (data_valid_dmat, 'valid_0')],
        'evals': [(data_train_dmat, 'train')],
        'maximize': False,
        # 'early_stopping_rounds': 100,
        'verbose_eval': True}
    mdl_bst = xgb.train(**train_params)

    logging.info('Saving model')
    # not use save_model mothod because it cannot dump best_iteration etc.
    cPickle.dump(mdl_bst, open(model_file, 'wb'))

    feat_imp = mdl_bst.get_score(importance_type='gain').items()
    print sorted(feat_imp, key=itemgetter(1), reverse=True)[0:10]


def test():

    resf = open('test_res.xgb', 'w')
    data_test_dmat = xgb.DMatrix(testf)

    # init gbt
    mdl_bst = cPickle.load(open(model_file, 'rb'))
    mdl_bst.set_param('nthread', 1)
    mdl_bst.set_param('eval_metric', 'rmse')

    test_metric = mdl_bst.eval_set([(data_test_dmat, 'test_0')])
    print test_metric.split(':')[-1], float(test_metric.split(':')[-1]) ** 2 / 2.
    pred_res = mdl_bst.predict(
        data_test_dmat,
        ntree_limit=mdl_bst.best_iteration)
    for i in pred_res:
        print >> resf, i
    resf.close()


def transform(ifilen, ofilen):
    ofile = open(ofilen, 'w')
    data_dmat = xgb.DMatrix(ifilen)

    # init gbt
    mdl_bst = cPickle.load(open(model_file, 'rb'))
    mdl_bst.set_param('nthread', 1)

    pred_res = mdl_bst.predict(
        data_dmat,
        ntree_limit=mdl_bst.best_iteration,
        pred_leaf=True)
    for l, i in zip(data_dmat.get_label(), pred_res):
        print >> ofile, str(l) + ' ' + ' '.join(map(str, i))
    ofile.close()


if __name__ == '__main__':
    #train()
    #test()
    transform(trainf, train_leaf_feat)
    transform(testf, test_leaf_feat)
    transform(predf, pred_leaf_feat)
