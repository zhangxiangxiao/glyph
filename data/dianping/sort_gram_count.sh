#!/bin/bash

# Sort distributed grams file
# Copyright 2016 Xiang Zhang
#
# Usage: bash sort_gram_count.sh [input_directory] [output_directory] [temporary] [memory]

set -x;
set -e;

for file in $1/*.csv; do
    sort -S ${4:-50%} -t ',' -k1,1 -T ${3:-/scratch} $file > $2/`basename $file`
done;
