# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import re
import sys
import argparse

import pandas
import pymongo
from clint.textui import puts, progress

from common import transliterate_like_rails
from finallyaresult import strip_deputy_name


def main(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('-H', '--host', type=unicode, default='localhost')
    parser.add_argument('-P', '--port', type=int, default=27017)
    parser.add_argument('-d', '--database', type=unicode, default='retorica_development')

    args = parser.parse_args(argv[1:])

    mongo = pymongo.MongoClient(args.host, args.port)
    database = getattr(mongo, args.database)

    speeches = database.discursos.find()
    speeches = progress.bar(speeches, expected_size=database.discursos.count())

    for s in speeches:
        s['author_clean'] = transliterate_like_rails(strip_deputy_name(s['autor']))

        deputy = database.deputados.find_one({
            'clean_name': s['author_clean'],
        })

        x = database.discursos.update({'_id': s['_id']}, {
            '$set': {
                'author_clean': s['author_clean'],
                'deputy_id': deputy['_id'] if deputy else None,
            },
        })


if __name__ == '__main__':
    sys.exit(main(sys.argv) or 0)
