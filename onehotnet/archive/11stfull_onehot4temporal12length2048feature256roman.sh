#!/bin/bash

# Archived program command-line for experiment
# Copyright 2016 Xiang Zhang
#
# Usage: bash {this_file} [additional_options]

set -x;
set -e;

qlua main.lua -train_data_file data/11st/sentiment/full_train_rr.t7b -test_data_file data/11st/sentiment/full_test_rr.t7b -driver_location models/11stfull/onehot4temporal12length2048feature256roman "$@";
