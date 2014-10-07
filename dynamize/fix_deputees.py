# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import re
import sys
import argparse
import unicodedata

import pandas
import pymongo
import unihandecode
from clint.textui import puts, progress


def transliterate_like_rails(string):
    # XXX FIXME Is this enough like Rails' version?
    # Convert to ASCII and back to Unicode
    string = unicodedata.normalize('NFKC', string)
    string = unihandecode.unidecode(string)
    return string.encode('ascii', 'replace').decode('utf-8')


def main(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('-H', '--host', type=unicode, default='localhost')
    parser.add_argument('-P', '--port', type=int, default=27017)
    parser.add_argument('-d', '--database', type=unicode, default='retorica_development')

    args = parser.parse_args(argv[1:])

    mongo = pymongo.MongoClient(args.host, args.port)
    database = getattr(mongo, args.database)
    collection = getattr(database, 'deputados')

    deputees = collection.find()
    deputees = progress.bar(speeches, expected_size=collection.count())

    for d in deputees:
        collection.update({'_id': d['_id']}, {
            '$set': {
                'clean_name': transliterate_like_rails(d['nome_parlamentar']),
            },
        })


if __name__ == '__main__':
    sys.exit(main(sys.argv) or 0)
