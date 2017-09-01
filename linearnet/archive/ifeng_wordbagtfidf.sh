#!/bin/bash

# Archived program command-line for experiment
# Copyright 2016 Xiang Zhang
#
# Usage: bash {this_file} [additional_options]

set -x;
set -e;

th main.lua -driver_location models/ifeng/wordbagtfidf -train_data_file data/ifeng/topic/train_wordbagtfidf.t7b -test_data_file data/ifeng/topic/test_wordbagtfidf.t7b "$@";
