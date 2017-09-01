#!/bin/bash

# Archived program command-line for experiment
# Copyright 2016 Xiang Zhang
#
# Usage: bash {this_file} [additional_options]

set -x;
set -e;

qlua main.lua -driver_location models/rakutenfull/temporal12length512feature256 -train_data_file data/rakuten/sentiment/full_train_code.t7b -test_data_file data/rakuten/sentiment/full_test_code.t7b "$@";
