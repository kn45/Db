#!/bin/bash

data_raw=data/d_train_20180102.csv
data_all=data/data_all.tsv

data_sub=data/d_test_A_20180102.csv
data_pred=data/data_pred.tsv

python 1_GetTsv.py $data_raw $data_all label
python 1_GetTsv.py $data_sub $data_pred nolabel
