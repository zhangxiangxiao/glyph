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

INPUT = '../data/ifeng/article'
OUTPUT = '../data/ifeng/topic/news.csv'

# Classes
# 1: Mainlaind China Politics
# 2: International
# 3: Taiwan, Hong Kong and Macau Politics
# 4: Military
# 5: Society
CLASSES = {'11528': 1, '11574': 2, '11490': 3, '7609': 3, '4550': 4, '7837': 5}

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
    for prefix in CLASSES:
        files = glob.glob(INPUT + '/' + prefix + '_*.json.xz')
        index = CLASSES[prefix]
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
