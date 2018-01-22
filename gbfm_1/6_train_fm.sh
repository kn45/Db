#!/bin/bash


~/libfm/bin/libFM \
-task r \
-train feat/trnvld_leaf.libfm \
-test feat/test_leaf.libfm \
-dim ’1,1,256’ \
#-method sgd -learn_rate 0.01 \
-iter 100

