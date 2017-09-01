#!/bin/bash

# Archived program command-line for experiment
# Copyright 2016 Xiang Zhang
#
# Usage: bash {this_file} [additional_options]

set -x;
set -e;

qlua main.lua -driver_location models/jdfull/temporal8length486feature256 -driver_variation small -train_data_file data/jd/sentiment/full_train_code.t7b -test_data_file data/jd/sentiment/full_test_code.t7b "$@";
