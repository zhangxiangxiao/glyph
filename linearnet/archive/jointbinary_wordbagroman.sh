#!/bin/bash

# Archived program command-line for experiment
# Copyright 2016 Xiang Zhang
#
# Usage: bash {this_file} [additional_options]

set -x;
set -e;

th main.lua -driver_location models/jointbinary/wordbagroman -train_data_file data/joint/binary_train_roman_wordbag.t7b -test_data_file data/joint/binary_test_roman_wordbag.t7b "$@";
