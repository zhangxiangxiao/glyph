#!/usr/bin/python3

'''
Convert Chinese datasets to Pinyin format
Copyright 2016 Xiang Zhang

Usage: python3 construct_pinyin.py -i [input] -o [output]
'''

#Input file
INPUT = '../data/dianping/train.csv'
#Output file
OUTPUT = '../data/dianping/train_pinyin.csv'

import argparse
import csv
import pypinyin
import unidecode

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

    convertPinyin()

# Convert the text in Chinese to pintin
def convertPinyin():
    # Open the files
    ifd = open(INPUT, encoding = 'utf-8', newline = '')
    ofd = open(OUTPUT, 'w', encoding = 'utf-8', newline = '')
    reader = csv.reader(ifd, quoting = csv.QUOTE_ALL)
    writer = csv.writer(ofd, quoting = csv.QUOTE_ALL, lineterminator = '\n')
    # Loop over the csv rows
    n = 0
    for row in reader:
        new_row = list()
        new_row.append(row[0])
        for i in range(1, len(row)):
            new_row.append(' '.join(map(
                str.strip,
                map(lambda s: s.replace('\n', '\\n'),
                    map(unidecode.unidecode,
                        pypinyin.lazy_pinyin(
                            row[i], style = pypinyin.TONE2))))))
        writer.writerow(new_row)
        n = n + 1
        if n % 1000 == 0:
            print('\rProcessing line: {}'.format(n), end = '')
    print('\rProcessed lines: {}'.format(n))

if __name__ == '__main__':
    main()
