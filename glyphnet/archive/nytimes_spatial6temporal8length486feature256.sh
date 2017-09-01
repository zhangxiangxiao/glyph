#!/bin/bash

# Archived program command-line for experiment
# Copyright 2016 Xiang Zhang
#
# Usage: bash {this_file} [additional_options]

set -x;
set -e;

qlua main.lua -driver_variation small -driver_location models/nytimes/spatial6temporal8length486feature256 -train_data_file data/nytimes/topic/train_code.t7b -test_data_file data/nytimes/topic/test_code.t7b "$@";
