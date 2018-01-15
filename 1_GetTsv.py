#!/usr/bin/env python

import csv
import sys

data_in = sys.argv[1]
data_out = sys.argv[2]
mode = sys.argv[3]


def raw2tsv(fields):
    if mode == 'nolabel':
        label = ['']
        features = fields
    else:
        label = fields[-1:]
        features = fields[:-1]
    return '\t'.join(label + features)


def run():
    with open(data_in) as fi_csv, open(data_out, 'w') as fo:
        fi = csv.reader(fi_csv)
        fi.next()  # skip col name
        for fields in fi:
            print >> fo, raw2tsv(fields)


if __name__ == '__main__':
    run()
