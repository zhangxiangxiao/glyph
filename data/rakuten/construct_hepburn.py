#!/usr/bin/python3

'''
Convert Japanese datasets to Hepburn Romanization
Copyright 2016 Xiang Zhang

Usage: python3 construct_hepburn.py -i [input] -o [output]
'''

# Input file
INPUT = '../data/rakuten/sentiment/full_train.csv'
# Output file
OUTPUT = '../data/rakuten/sentiment/full_train_hepburn.csv'

import argparse
import csv
import MeCab
import romkan
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

    mecab = MeCab.Tagger()

    convertRoman(mecab)

def romanizeText(mecab, text):
    parsed = mecab.parse(text)
    result = list()
    for token in parsed.split('\n'):
        splitted = token.split('\t')
        if len(splitted) == 2:
            word = splitted[0]
            features = splitted[1].split(',')
            if len(features) > 7 and features[7] != '*':
                result.append(romkan.to_hepburn(features[7]))
            else:
                result.append(word)
    return result

# Convert the text in Chinese to pintin
def convertRoman(mecab):
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
                        romanizeText(mecab, row[i]))))))
        writer.writerow(new_row)
        n = n + 1
        if n % 1000 == 0:
            print('\rProcessing line: {}'.format(n), end = '')
    print('\rProcessed lines: {}'.format(n))

if __name__ == '__main__':
    main()
