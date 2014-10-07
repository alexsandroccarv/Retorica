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


def strip_deputy_name(name):
    # strip AKIRA OTSUBO (PRESIDENTE)
    name = re.sub(r'\s*\([^\)]+\)\s*$', '', name)

    # strip ALGUEM, ALGUMA COISA
    name = re.sub(r'\s*,.*$', '', name)

    # strip ALGUEM - PARLAMENTAR JOVEM
    # but don't touch AKIRA-TO
    name = re.sub(r'\s+-.*$', '', name)

    return name


def find_deputy_by_transliterated_name(collection, name):
    # XXX FIXME should be atomic!
    deputy = collection.find_one({'nome_parlamentar': name})

    if deputy is None:
        # Try again with some transliteration
        # This happens to a guy named ANDRÉ -something
        name = transliterate_like_rails(name)
        deputy = collection.find_one({'nome_parlamentar': name})

    return deputy


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
