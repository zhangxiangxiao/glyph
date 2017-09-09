#!/usr/bin/python3

'''
Convert Korean datasets to Revised Romanization of Korean (RR, MC2000)
Copyright 2016 Xiang Zhang

Usage: python3 construct_hepburn.py -i [input] -o [output]
'''

# Input file
INPUT = '../data/11st/sentiment/full_train.csv'
# Output file
OUTPUT = '../data/11st/sentiment/full_train_rr.csv'

import argparse
import csv
import hanja
import unidecode

# Hangul romanization libraries
from hangul_romanize import Transliter
from hangul_romanize.rule import academic

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

    transliter = Transliter(academic)

    convertRoman(transliter)

def romanizeText(transliter, text):
    text = text.strip()
    if text != '':
        hangul_text = hanja.translate(text, 'substitution')
        return transliter.translit(hangul_text)
    return text

# Convert the text in Chinese to pintin
def convertRoman(transliter):
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
            new_row.append(unidecode.unidecode(romanizeText(
                        transliter, row[i])).strip().replace('\n','\\n'))
        writer.writerow(new_row)
        n = n + 1
        if n % 1000 == 0:
            print('\rProcessing line: {}'.format(n), end = '')
    print('\rProcessed lines: {}'.format(n))

if __name__ == '__main__':
    main()
