#!/bin/bash

# Archived program command-line for experiment
# Copyright 2016 Xiang Zhang
#
# Usage: bash {this_file} [additional_options]

set -x;
set -e;

qlua main.lua -train_data_file data/chinanews/topic/train_code.t7b -test_data_file data/chinanews/topic/test_code.t7b -driver_location models/chinanews/temporal8length486feature256 -driver_variation small "$@";
