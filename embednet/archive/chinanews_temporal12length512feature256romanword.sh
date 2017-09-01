#!/bin/bash

# Archived program command-line for experiment
# Copyright 2016 Xiang Zhang
#
# Usage: bash {this_file} [additional_options]

set -x;
set -e;

qlua main.lua -driver_location models/chinanews/temporal12length512feature256romanword -driver_dimension 200002 -train_data_file data/chinanews/topic/train_pinyin_word_limit.t7b -train_data_replace 200002 -test_data_file data/chinanews/topic/test_pinyin_word_limit.t7b -test_data_replace 200002 "$@";
