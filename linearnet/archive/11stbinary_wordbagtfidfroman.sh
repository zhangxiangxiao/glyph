#!/bin/bash

# Archived program command-line for experiment
# Copyright 2016 Xiang Zhang
#
# Usage: bash {this_file} [additional_options]

set -x;
set -e;

th main.lua -driver_location models/11stbinary/wordbagtfidfroman -train_data_file data/11st/sentiment/binary_train_rr_wordbagtfidf.t7b -test_data_file data/11st/sentiment/binary_test_rr_wordbagtfidf.t7b "$@";
