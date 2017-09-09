#!/usr/bin/python3

'''
Remove duplication from csv format file
Copyright 2015 Xiang Zhang

Usage: python3 remove_duplication.py -i [input] -o [output]
'''

# Python 3 compatibility
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# Input file
INPUT = '../data/dianping/reviews_nonull.csv'
# Output file
OUTPUT = '../data/dianping/reviews_nodup.csv'

import argparse
import csv

# Main program
def main():
    global INPUT
    global OUTPUT

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help = 'Input file', default = INPUT)
    parser.add_argument(
        '-o', '--output', help = 'Output file', default = OUTPUT)

    args = parser.parse_args()

    INPUT = args.input
    OUTPUT = args.output

    removeDuplicate()

# Deduplicate the text using python set
def removeDuplicate():
    # Open the files
    ifd = open(INPUT, newline = '', encoding = 'utf-8')
    ofd = open(OUTPUT, 'w', newline = '', encoding = 'utf-8')
    reader = csv.reader(ifd, quoting = csv.QUOTE_ALL)
    writer = csv.writer(ofd, quoting = csv.QUOTE_ALL, lineterminator = '\n')
    # Loop over the csv rows
    n = 0
    valid = 0
    s = set()
    for row in reader:
        line = ' '.join(row[1:])
        n = n + 1
        if line not in s:
            valid = valid + 1
            s.add(line)
            writer.writerow(row)
        if n % 10000 == 0:
            print('\rProcessing line: {}, valid: {}'.format(n, valid), end = '')
    print('\rProcessed lines: {}, valid: {}'.format(n, valid))

if __name__ == '__main__':
    main()
