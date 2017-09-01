#!/bin/bash

# Archived program command-line for experiment
# Copyright 2016 Xiang Zhang
#
# Usage: bash {this_file} [additional_options]

set -x;
set -e;

qlua main.lua -driver_variation small -train_data_file data/joint/binary_train_roman.t7b -test_data_file data/joint/binary_test_roman.t7b -driver_steps 400000 -driver_location models/jointbinary/onehot4temporal8length1944feature256roman "$@";
