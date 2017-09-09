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
import re
import urllib.parse

INPUT = '../data/nytimes/article'
OUTPUT = '../data/nytimes/topic/news.csv'
CLASS = '../data/nytimes/topic/class.csv'

def main():
    global INPUT
    global OUTPUT
    global CLASS

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input', help = 'Input file directory', default = INPUT)
    parser.add_argument(
        '-o', '--output', help = 'Output file', default = OUTPUT)
    parser.add_argument(
        '-c', '--classes', help = 'Class file', default = CLASS)

    args = parser.parse_args()

    INPUT = args.input
    OUTPUT = args.output
    CLASS = args.classes

    createData()

def createData():
    # Open the category file
    classes = dict()
    count = 0
    # Open the output file
    ofd = open(OUTPUT, 'w', newline = '', encoding = 'utf-8')
    writer = csv.writer(ofd, quoting = csv.QUOTE_ALL, lineterminator = '\n')
    # Grab the files
    files = glob.glob(INPUT + '/*.json.xz')
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
                url = news.get('url', '')
                if url != '':
                    path = urllib.parse.urlparse(url).path
                    start_match = re.match(r'/\d\d\d\d/\d\d/\d\d/', path)
                    end_match = re.match(r'/\d\d\d\d/\d\d/\d\d/[^/]+', path)
                    if start_match != None and end_match != None:
                        classname = path[start_match.end():end_match.end()]
                        if classes.get(classname, None) == None:
                            classes[classname] = count + 1
                            count = count + 1
                        index = classes[classname]
                        writer.writerow([index, title.replace('\n', '\\n'),
                                         abstract.replace('\n', '\\n')])
                n = n + 1
            ifd.close()
        except Exception as e:
            print('Exception (ignored): {}'.format(e))
    ofd.close()
    # Open the class file
    cfd = open(CLASS, 'w', newline = '', encoding = 'utf-8')
    class_writer = csv.writer(
        cfd, quoting = csv.QUOTE_ALL, lineterminator = '\n')
    for key in classes:
        class_writer.writerow([classes[key], key])
    cfd.close()

if __name__ == '__main__':
    main()
