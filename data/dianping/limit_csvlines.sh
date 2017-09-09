#!/bin/bash

# Limit csv files to designated number of lines
# Copyright 2015 Xiang Zhang
#
# Usage: bash limit_csvlines.sh [input] [output] [limit]

set -x;
set -e;

head -n ${3:-1000001} $1 > $2;
