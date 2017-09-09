#!/bin/bash

# Split lines in a text file
# Copyright 2017 Xiang Zhang
#
# Usage: bash split_lines.sh [lines] [input] [output_prefix]
#
# Note: .txt postfix will be automatically added.

set -x;
set -e;

split -d -a 1 --additional-suffix=.txt -l $1 $2 $3;
