#!/bin/bash

# Archived program command-line for experiment
# Copyright 2017 Xiang Zhang
#
# Usage: bash {this_file} [additional_options]

set -x;
set -e;

LOCATION=models/amazonfull/wordbigram_evaluation;
TRAIN_DATA=data/amazon/full_train_wordtoken_shuffle_split_0.txt;
TEST_DATA=data/amazon/full_train_wordtoken_shuffle_split_1.txt;

fasttext supervised -input $TRAIN_DATA -output $LOCATION/model_2 -dim 10 -lr 0.1 -wordNgrams 2 -minCount 1 -bucket 10000000 -epoch 2 -thread 10;
fasttext test $LOCATION/model_2.bin $TEST_DATA;
fasttext supervised -input $TRAIN_DATA -output $LOCATION/model_5 -dim 10 -lr 0.1 -wordNgrams 2 -minCount 1 -bucket 10000000 -epoch 5 -thread 10;
fasttext test $LOCATION/model_5.bin $TEST_DATA;
fasttext supervised -input $TRAIN_DATA -output $LOCATION/model_10 -dim 10 -lr 0.1 -wordNgrams 2 -minCount 1 -bucket 10000000 -epoch 10 -thread 10;
fasttext test $LOCATION/model_10.bin $TEST_DATA;
