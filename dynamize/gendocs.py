# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import sys
import json
import argparse
import datetime

import pymongo

from clint.textui import puts, progress


def mkdate(s):
    try:
        return datetime.datetime.strptime(s, '%Y-%m-%d')
    except Exception, e:
        puts("Could not parse date `{0}'. Apparently not in the format Y-m-d: {1}".format(s, e))


def main(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('-H', '--host', type=unicode, default='localhost')
    parser.add_argument('-P', '--port', type=int, default=27017)
    parser.add_argument('-d', '--database', type=unicode, default='retorica_development')
    parser.add_argument('--initial-term', type=mkdate, default=None,
                        help=('ignore all documents previously to this date (in the Y-m-d format)'))
    parser.add_argument('--final-term', type=mkdate, default=None,
                        help=('ignore all documents after this date (in the Y-m-d format)'))

    parser.add_argument('outfile', type=argparse.FileType(mode='w'))

    args = parser.parse_args(argv[1:])

    client = pymongo.MongoClient(args.host, args.port)
    database = getattr(client, args.database)

    lookup = {
        'conteudo_stemmed': {'$exists': True},
    }

    dt_lookup = {}

    if args.initial_term:
        dt_lookup['$gte'] = args.initial_term
    if args.final_term:
        dt_lookup['$lte'] = args.final_term

    if dt_lookup:
        lookup['proferido_em'] = dt_lookup

    database.discursos.ensure_index('autor')

    documents = database.discursos.find(lookup)

    documents = documents.sort('autor')

    for d in progress.bar(documents, expected_size=documents.count()):
        args.outfile.write(json.dumps([d['autor'], d['conteudo_stemmed']]) + '\n')


if __name__ == '__main__':
    sys.exit(main(sys.argv) or 0)
