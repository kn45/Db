#!/bin/bash


~/libfm/bin/libFM \
-task r \
-train feat/trnvld_leaf.libfm \
-test feat/test_leaf.libfm \
-dim ’1,1,8’ \
-iter 1000
#-method sgd -learn_rate 0.1 -regular ’0,0,0.01’ -init_stdev 0.1 \

