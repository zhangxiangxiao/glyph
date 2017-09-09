#!/usr/bin/python3

'''
Create data from list of LZMA compressed archives of news articles
Copyright 2016 Xiang Zhang

Usage: python3 construct_topic.py -i [input directory] -o [output file]
'''

import argparse
import csv
import glob
import json
import lzma

INPUT = '../data/chinanews/article'
OUTPUT = '../data/chinanews/topic/news.csv'
CATEGORY_FILE = '../data/chinanews/category/category.json'

def main():
    global INPUT
    global OUTPUT
    global CATEGORY_FILE

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input', help = 'Input file directory', default = INPUT)
    parser.add_argument(
        '-o', '--output', help = 'Output file', default = OUTPUT)
    parser.add_argument(
        '-c', '--category', help = 'Category file', default = CATEGORY_FILE)

    args = parser.parse_args()

    INPUT = args.input
    OUTPUT = args.output
    CATEGORY_FILE = args.category

    createData()

def createData():
    # Open the category file
    classes = dict()
    cfd = open(CATEGORY_FILE, encoding = 'utf-8')
    i = 1
    for line in cfd:
        category = json.loads(line)
        classes[category['code']] = i
        i = i + 1
    # Open the output file
    ofd = open(OUTPUT, 'w', newline = '', encoding = 'utf-8')
    writer = csv.writer(ofd, quoting = csv.QUOTE_ALL, lineterminator = '\n')
    # Grab the files
    for prefix in classes:
        files = glob.glob(INPUT + '/' + prefix + '_*.json.xz')
        index = classes[prefix]
        n = 0
        filecount = 0
        for filename in files:
            filecount = filecount + 1
            print('Processing file {}/{}: {}. Processed items {}.'.format(
                    filecount, len(files), filename, n))
            try:
                ifd = lzma.open(filename, 'rt', encoding = 'utf-8')
                for line in ifd:
                    news = json.loads(line)
                    title = news.get('title', '')
                    content = news.get('content', list())
                    abstract = ''
                    if len(content) > 0:
                        abstract = content[0]
                    n = n + 1
                    writer.writerow([index, title.replace('\n', '\\n'),
                                     abstract.replace('\n', '\\n')])
                ifd.close()
            except Exception as e:
                print('Exception (ignored): {}'.format(e))
    ofd.close()

if __name__ == '__main__':
    main()
