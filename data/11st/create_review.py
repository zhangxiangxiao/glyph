#!/usr/bin/python3

'''
Create data from list of LZMA compressed archives of reviews
Copyright 2016 Xiang Zhang

Usage: python3 create_review.py -i [input file pattern] -o [output file]
'''

import argparse
import csv
import glob
import json
import lzma

INPUT = '../data/11st/review/*.json.xz'
OUTPUT = '../data/11st/sentiment/review.csv'

def main():
    global INPUT
    global OUTPUT

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input', help = 'Input file pattern', default = INPUT)
    parser.add_argument(
        '-o', '--output', help = 'Output file', default = OUTPUT)

    args = parser.parse_args()

    INPUT = args.input
    OUTPUT = args.output

    createData()

def createData():
    # Open the output file
    ofd = open(OUTPUT, 'w', newline = '', encoding = 'utf-8')
    writer = csv.writer(ofd, quoting = csv.QUOTE_ALL, lineterminator = '\n')
    # Grab the files
    files = glob.glob(INPUT)
    n = 0
    filecount = 0
    for filename in files:
        filecount = filecount + 1
        print('Processing file {}/{}: {}. Processed items {}.'.format(
                filecount, len(files), filename, n))
        try:
            ifd = lzma.open(filename, 'rt', encoding = 'utf-8')
            for line in ifd:
                review = json.loads(line)
                star = review.get('star', '')
                title = review.get('title', '')
                content = review.get('content', '')
                if star != '':
                    n = n + 1
                    writer.writerow([star, title.replace('\n', '\\n'),
                                     content.replace('\n', '\\n')])
            ifd.close()
        except Exception as e:
            print('Exception (ignored): {}'.format(e))
    ofd.close()

if __name__ == '__main__':
    main()
