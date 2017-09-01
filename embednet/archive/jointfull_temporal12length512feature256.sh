#!/bin/bash

# Archived program command-line for experiment
# Copyright 2016 Xiang Zhang
#
# Usage: bash {this_file} [additional_options]

set -x;
set -e;

qlua main.lua -train_data_file data/joint/full_train_code.t7b -test_data_file data/joint/full_test_code.t7b -driver_location models/jointfull/temporal12length512feature256 -driver_steps 400000 "$@";
