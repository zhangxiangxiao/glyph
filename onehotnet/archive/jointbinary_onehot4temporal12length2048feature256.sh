#!/bin/bash

# Archived program command-line for experiment
# Copyright 2016 Xiang Zhang
#
# Usage: bash {this_file} [additional_options]

set -x;
set -e;

qlua main.lua -train_data_file data/joint/binary_train.t7b -test_data_file data/joint/binary_test.t7b -driver_steps 400000 -driver_location models/jointbinary/onehot4temporal12length2048feature256 "$@";
