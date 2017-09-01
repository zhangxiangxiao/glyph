#!/bin/bash

# Archived program command-line for experiment
# Copyright 2016 Xiang Zhang
#
# Usage: bash {this_file} [additional_options]

set -x;
set -e;

th main.lua -driver_location models/dianping/chargram -train_data_file data/dianping/train_chargram.t7b -test_data_file data/dianping/test_chargram.t7b -model_size 1000001 "$@";
