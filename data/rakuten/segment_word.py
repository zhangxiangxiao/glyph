#!/usr/bin/python3

'''
Convert Japanese datasets to Index of Words
Copyright 2016 Xiang Zhang

Usage: python3 construct_pinyin.py -i [input] -l [list] -o [output] [-r]
'''

#Input file
INPUT = '../data/rakuten/sentiment/full_train.csv'
#Output file
OUTPUT = '../data/rakuten/sentiment/full_train_word.csv'
# List file
LIST = '../data/rakuten/sentiment/full_train_word_list.csv'
# Read already defined word list
READ = False

import argparse
import csv
import MeCab

# Main program
def main():
    global INPUT
    global OUTPUT
    global LIST

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help = 'Input file', default = INPUT)
    parser.add_argument(
        '-o', '--output', help = 'Output file', default = OUTPUT)
    parser.add_argument('-l', '--list', help = 'Word list file', default = LIST)
    parser.add_argument(
        '-r', '--read', help = 'Read from list file', action = 'store_true')

    args = parser.parse_args()

    INPUT = args.input
    OUTPUT = args.output
    LIST = args.list
    READ = args.read

    if READ:
        print('Reading word index')
        word_index = readWords()
    else:
        print('Counting words')
        word_count, word_freq = segmentWords()
        print('Sorting words by count')
        word_index = sortWords(word_count, word_freq)
    print('Constructing word index output')
    convertWords(word_index)

# Read from pre-existing word list
def readWords():
    # Open the files
    ifd = open(LIST, encoding = 'utf-8', newline = '')
    reader = csv.reader(ifd, quoting = csv.QUOTE_ALL)
    # Loop over the csv rows
    word_index = dict()
    n = 0
    for row in reader:
        word = row[0].replace('\\n', '\n')
        word_index[word] = n + 1
        n = n + 1
        if n % 1000 == 0:
            print('\rProcessing line: {}'.format(n), end = '')
    print('\rProcessed lines: {}'.format(n))
    return word_index

# Segment the text in Chinese
def segmentWords():
    mecab = MeCab.Tagger()
    # Open the files
    ifd = open(INPUT, encoding = 'utf-8', newline = '')
    reader = csv.reader(ifd, quoting = csv.QUOTE_ALL)
    # Loop over the csv rows
    word_count = dict()
    word_freq = dict()
    n = 0
    for row in reader:
        field_set = set()
        for i in range(1, len(row)):
            field = row[i].replace('\\n', '\n')
            field_list = list()
            parsed_result = mecab.parse(field)
            for token in parsed_result.split('\n'):
                splitted_token = token.split('\t')
                if len(splitted_token) == 2:
                    word = splitted_token[0]
                    field_list.append(word)
            for word in field_list:
                word_count[word] = word_count.get(word, 0) + 1
                if word not in field_set:
                    field_set.add(word)
                    word_freq[word] = word_freq.get(word, 0) + 1
        n = n + 1
        if n % 1000 == 0:
            print('\rProcessing line: {}'.format(n), end = '')
    print('\rProcessed lines: {}'.format(n))
    ifd.close()
    # Normalizing word frequency
    for word in word_freq:
        word_freq[word] = float(word_freq[word]) / float(n)
    return word_count, word_freq

# Sort words for a given count dictionary object
def sortWords(word_count, word_freq):
    # Sort the words
    word_list = sorted(
        word_count, key = lambda word: word_count[word], reverse = True)
    # Open the files
    ofd = open(LIST, 'w', encoding = 'utf-8', newline = '')
    writer = csv.writer(ofd, quoting = csv.QUOTE_ALL, lineterminator = '\n')
    # Loop over all the words
    word_index = dict()
    n = 0
    for i in range(len(word_list)):
        word = word_list[i]
        row = [word.replace('\n', '\\n'), str(word_count[word]),
               str(word_freq[word])]
        writer.writerow(row)
        word_index[word] = i + 1
        n = n + 1
        if n % 1000 == 0:
            print('\rProcessing word: {}'.format(n), end = '')
    print('\rProcessed words: {}'.format(n))
    ofd.close()
    return word_index

# Convert the text in Chinese to word list
def convertWords(word_index):
    mecab = MeCab.Tagger()
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
            field = row[i].replace('\\n', '\n')
            field_list = list()
            parsed_result = mecab.parse(field)
            for token in parsed_result.split('\n'):
                splitted_token = token.split('\t')
                if len(splitted_token) == 2:
                    word = splitted_token[0]
                    field_list.append(word)
            new_row.append(' '.join(map(
                str, map(lambda word: word_index.get(word, len(word_index) + 1),
                         field_list))))
        writer.writerow(new_row)
        n = n + 1
        if n % 1000 == 0:
            print('\rProcessing line: {}'.format(n), end = '')
    print('\rProcessed lines: {}'.format(n))
    ifd.close()
    ofd.close()

if __name__ == '__main__':
    main()
