# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import sys
import numpy
import pandas
import argparse


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('thetasfile', type=argparse.FileType('r'))
    parser.add_argument('musfile', type=argparse.FileType('r'))

    args = parser.parse_args(argv[1:])

    thetas = pandas.read_csv(args.thetasfile, index_col=0, encoding='utf-8')

    author_topics = thetas.idxmax(axis=1)

    # Calculate the proportion table for the input
    t = []
    for i in range(thetas.shape[1]):
        row = thetas.irow(i)
        t.append(row / numpy.sum(row))

    propthetas = pandas.DataFrame(t)

    result = []
    for i, topic_idx in enumerate(propthetas.idxmax(axis=1)):
        row = propthetas.irow(i)
        enfase = row[topic_idx]
        result.append((topic_idx, enfase))

    topics = pandas.DataFrame(result, index=propthetas.index)
    topics.to_csv('result.csv', encoding='utf-8')

    mus = pandas.read_csv(args.musfile, index_col=0, encoding='utf-8')

    topicwords = []
    for i in range(mus.shape[1]):
        words = mus.icol(i).sort(ascending=False, inplace=False)
        topicwords.append([w for (w, e) in words[:30].iteritems()])

    words = pandas.DataFrame(topicwords)
    words.to_csv('words.csv', header=False, index=False, encoding='utf-8')


if __name__ == '__main__':
    sys.exit(main(sys.argv) or 0)
