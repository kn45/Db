#!/bin/bash

cat feat/test_feature.libsvm | cut -d' ' -f1 > test_label
cat test_label | awk '{print exp($1)}' > test_label.tmp
cat test_res | awk '{print exp($1)}' > test_res.tmp
paste test_label.tmp test_res.tmp | awk 'BEGIN{s=0;n=0}{s+=($1-$2)^2;n+=1}END{print s/n}'
