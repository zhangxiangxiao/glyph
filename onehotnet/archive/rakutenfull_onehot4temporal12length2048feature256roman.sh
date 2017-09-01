#!/bin/bash

# Archived program command-line for experiment
# Copyright 2016 Xiang Zhang
#
# Usage: bash {this_file} [additional_options]

set -x;
set -e;

qlua main.lua -train_data_file data/rakuten/sentiment/full_train_hepburn.t7b -test_data_file data/rakuten/sentiment/full_test_hepburn.t7b -driver_location models/rakutenfull/onehot4temporal12length2048feature256roman "$@";
