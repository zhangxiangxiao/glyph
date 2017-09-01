#!/bin/bash

# Archived program command-line for experiment
# Copyright 2017 Xiang Zhang
#
# Usage: bash {this_file} [additional_options]

set -x;
set -e;

LOCATION=models/chinanews/wordbigramroman_tuned;
TRAIN_DATA=data/chinanews/topic/train_pinyin_wordtoken_shuffle.txt;
TEST_DATA=data/chinanews/topic/test_pinyin_wordtoken_shuffle.txt;

fasttext supervised -input $TRAIN_DATA -output $LOCATION/model -dim 10 -lr 0.1 -wordNgrams 2 -minCount 2 -bucket 10000000 -epoch 5 -thread 10;
fasttext test $LOCATION/model.bin $TRAIN_DATA;
fasttext test $LOCATION/model.bin $TEST_DATA;
