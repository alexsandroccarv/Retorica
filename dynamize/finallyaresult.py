# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import re
import os.path
import sys
import argparse
import unicodedata

import pandas
import pymongo
from clint.textui import puts

from common import transliterate_like_rails


def parameterize_like_rails(string, sep='-'):
    p = transliterate_like_rails(string)
    # Turn unwanted chars into the separator
    p = re.sub(r'[^a-z0-9\-_]+', sep, p)
    if sep:
        # No more than one separator in a row
        p = re.sub(r'(' + re.escape(sep) + '){2,}', sep, p)
        # Remove leading/trailing separators
        p = p.strip(sep)
    return p.lower()


def strip_deputy_name(name):
    # strip AKIRA OTSUBO (PRESIDENTE)
    name = re.sub(r'\s*\([^\)]+\)\s*$', '', name)

    # strip ALGUEM, ALGUMA COISA
    name = re.sub(r'\s*,.*$', '', name)

    # strip ALGUEM - PARLAMENTAR JOVEM
    # but don't touch AKIRA-TO
    name = re.sub(r'\s+-.*$', '', name)

    return name


def find_deputy_by_name(collection, name):
    clean_name = transliterate_like_rails(name)
    return deputy


def main(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('-H', '--host', type=unicode, default='localhost')
    parser.add_argument('-P', '--port', type=int, default=27017)
    parser.add_argument('-d', '--database', type=unicode, default='retorica_development')

    parser.add_argument('-t', '--title', type=unicode)

    parser.add_argument('results_folder', type=unicode, help='Path to the results folder')

    args = parser.parse_args(argv[1:])

    result = os.path.join(args.results_folder, 'result.csv')
    result = pandas.read_csv(result, index_col=0, encoding='utf-8')

    topics = os.path.join(args.results_folder, 'topics.csv')
    topics = pandas.read_csv(topics, header=None, index_col=0, encoding='utf-8')

    rs = os.path.join(args.results_folder, 'rs.csv')
    rs = pandas.read_csv(rs, index_col=0, encoding='utf-8')

    mongo = pymongo.MongoClient(args.host, args.port)
    database = getattr(mongo, args.database)

    dashboard_id = database.dashboards.insert({
        'slug': parameterize_like_rails(args.title),
        'title': args.title,
    })

    topic_idx_to_id = {}

    for row_idx in range(topics.shape[0]):
        row = topics.irow(row_idx)
        title = row.name
        observ = ignore = False

        if title.startswith('//'):
            title = title.lstrip('/')
            observ = True

        if title.startswith('^^'):
            title = title.lstrip('^')
            ignore = True

        topic_id = database.topics.insert({
            'title': title,
            'observ': observ,
            'ignore': ignore,
            'words': [w for w in row],
            'dashboard_id': dashboard_id,
        })

        topic_idx_to_id[row_idx] = topic_id

    for row_idx in range(result.shape[0]):
        row = result.irow(row_idx)
        name = row.name
        emphasis = row[1]
        topic_idx = int(row[0])
        topic_id = topic_idx_to_id[topic_idx]

        stripped_name = strip_deputy_name(name)
        clean_name= transliterate_like_rails(stripped_name)

        # XXX FIXME shouldn't this be atomic?
        deputy = database.deputados.find_one({
            '$or': [
                {'clean_name': clean_name},
                {'nome_parlamentar': stripped_name},
            ],
        })

        database.emphases.insert({
            'name': name,
            'clean_name': clean_name,
            'stripped_name': stripped_name,
            'emphasis': emphasis,
            'topic_id': topic_id,
            'deputado_id': deputy['_id'] if deputy else None,
        })


if __name__ == '__main__':
    sys.exit(main(sys.argv) or 0)
