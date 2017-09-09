#!/bin/bash

# Sort list of grams and cut the count
# Copyright 2016 Xiang Zhang
#
# Usage: bash sort_gram_list.sh [input] [output] [temporary] [memory]

set -x;
set -e;

sort -S ${4:-50%} -t ',' -k1,1nr -T ${3:-/scratch} $1 | cut -f 2- -d ',' > $2;
