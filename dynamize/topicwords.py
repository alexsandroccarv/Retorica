from __future__ import unicode_literals

import sys
import pandas
import argparse
from itertools import islice
from clint.textui import puts, indent


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--words', dest='words', type=int, default=30,
                        help='Number of words to print')
    parser.add_argument('input_file', type=argparse.FileType('r'))

    args = parser.parse_args()

    topicwords = pandas.read_csv(args.input_file, encoding='utf-8')

    idxpadding = ((max(map(len, topicwords.index)) % 4) + 2) * 4

    for col in topicwords.columns:
        words = topicwords[col].sort(ascending=False, inplace=False)
        #head = words.index[:args.words]
        head = list(islice(words.iteritems(), 0, args.words))
        head = ['{0},{1}'.format(*i) for i in head]
        puts("{0}{1}".format(col.ljust(idxpadding), ','.join(head)))


if __name__ == '__main__':
    main(sys.argv)
