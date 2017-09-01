#!/bin/bash

# Archived program command-line for experiment
# Copyright 2016 Xiang Zhang
#
# Usage: bash {this_file} [additional_options]

set -x;
set -e;

th main.lua -driver_location models/ifeng/wordgramroman -train_data_file data/ifeng/topic/train_pinyin_wordgram.t7b -test_data_file data/ifeng/topic/test_pinyin_wordgram.t7b -model_size 1000001 "$@";
