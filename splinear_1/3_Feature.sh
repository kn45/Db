#!/bin/bash

# Build feature using various ways from tsv file
# tsv -> feature

data_all=../data/data_all.tsv
data_trnvld=../data/data_trnvld.tsv
data_train=../data/data_train.tsv
data_valid=../data/data_valid.tsv
data_test=../data/data_test.tsv
data_pred=../data/data_pred.tsv

num_all=feat/all_num.libsvm
num_trnvld=feat/trnvld_num.libsvm
num_train=feat/train_num.libsvm
num_valid=feat/valid_num.libsvm
num_test=feat/test_num.libsvm
num_pred=feat/pred_num.libsvm

feat_all=feat/all_feature.libsvm
feat_trnvld=feat/trnvld_feature.libsvm
feat_train=feat/train_feature.libsvm
feat_valid=feat/valid_feature.libsvm
feat_test=feat/test_feature.libsvm
feat_pred=feat/pred_feature.libsvm

python 3_feat2num.py $data_all $num_all
python 3_feat2num.py $data_trnvld $num_trnvld
python 3_feat2num.py $data_train $num_train
python 3_feat2num.py $data_valid $num_valid
python 3_feat2num.py $data_test $num_test
python 3_feat2num.py $data_pred $num_pred

python 3_build_spliter.py $num_all

python 3_Feature.py $num_train $feat_train
python 3_Feature.py $num_valid $feat_valid
python 3_Feature.py $num_trnvld $feat_trnvld
python 3_Feature.py $num_test $feat_test
python 3_Feature.py $num_pred $feat_pred
python 3_Feature.py $num_all $feat_all
