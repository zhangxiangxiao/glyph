#!/bin/bash

# Sort comma-separated file starting from the second field
# Copyright 2016 Xiang Zhang
#
# Usage: bash sort_data.sh [input_file] [output_file] [temporary] [memory]

set -x;
set -e;

sort -S ${4:-50%} -t ',' -k2 -u -T ${3:-/scratch} $1 > $2;
