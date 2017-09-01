#!/bin/bash

# Archived program command-line for experiment
# Copyright 2016 Xiang Zhang
#
# Usage: bash {this_file} [additional_options]

set -x;
set -e;

qlua main.lua -driver_location models/rakutenbinary/temporal8length486feature256romanword -driver_variation small -driver_dimension 200002 -train_data_file data/rakuten/sentiment/binary_train_hepburn_word_limit.t7b -train_data_replace 200002 -test_data_file data/rakuten/sentiment/binary_test_hepburn_word_limit.t7b -test_data_replace 200002 "$@";
