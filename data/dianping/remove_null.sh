#!/bin/bash

# Remove NULL character from file
# Copyright 2015 Xiang Zhang
#
# Usage: bash remove_null.sh [input] [output]

set -x;
set -e;

tr -d '\000' < $1 > $2;
