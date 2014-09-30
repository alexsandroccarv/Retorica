# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import re
import pymongo
import sys
import unicodedata
import itertools
from StringIO import StringIO
from argparse import ArgumentParser

import nltk
import pyth.document
from pyth.plugins.plaintext.writer import PlaintextWriter
from clint.textui import puts, colored
from clint.textui import progress

# XXX we should use local relative imports, but we are not in a package yet
from rtfreader import CustomRtf15Reader as Rtf15Reader


def can_be_converted_to_text(p):
    """Should return `True` if the given `pyth.document.Paragraph` can be
    converted to text by the PlaintextWriter, `False` otherwise.

    ...But currently only returns `False` for images.
    """
    return not isinstance(p, pyth.document.Image)


def sanitize_rtf_document(doc):
    """Sanitize a `pyth.document.Document`, removing everything that can't be
    converted to plain text.

    WARNING! This method operates in place, changing the input *doc* and
    returning `None`.
    """
    for paragraph in doc.content:
        paragraph.content = filter(can_be_converted_to_text, paragraph.content)


def process_document(content):
    doc = StringIO(content)

    rtfdoc = Rtf15Reader.read(doc)

    # Remove non textual elements from the RTF document
    sanitize_rtf_document(rtfdoc)

    # Convert the RTF document to plain text
    plaintext = PlaintextWriter.write(rtfdoc).read().decode('utf-8')

    # Do our best to replace special characters (mostly accentuated chars)
    # with their corresponding transliterated simplified chars
    clean = unicodedata.normalize('NFKD', plaintext).encode('ascii', 'ignore').decode('utf-8')

    # Do our best to remove punctuation and stuff that don't compose words
    allowed_categories = set(('Lu', 'Ll', 'Nd', 'Zs'))

    filter_function = lambda c: c if unicodedata.category(c) in allowed_categories else '#'

    clean = ''.join(map(filter_function, clean)).replace('#', ' ')

    # We don't want no extra spaces
    clean = re.sub(r'\s+', ' ', clean).lower().strip()

    # Reduce words to their stemmed version
    stemmer = nltk.stem.snowball.PortugueseStemmer()

    stemmed = ' '.join(itertools.imap(stemmer.stem, clean.split()))

    return plaintext, stemmed


def main(argv):
    parser = ArgumentParser(prog='stemmer')

    parser.add_argument('-H', '--host', type=unicode, default='localhost')
    parser.add_argument('-P', '--port', type=int, default=27017)
    parser.add_argument('-d', '--database', type=unicode, default='retorica_development')

    args = parser.parse_args(argv[1:])

    mongo = pymongo.MongoClient(args.host, args.port)
    database = getattr(mongo, args.database)

    documents = database.discursos.find({'conteudo_stemmed': {'$exists': False}})

    puts("Processing {0} documents...".format(documents.count()))

    for d in progress.bar(documents, expected_size=documents.count()):

        try:
            plaintext, stemmed = process_document(d['conteudo'])
        except UnicodeDecodeError:
            puts(colored.red('Error at {0}'.format(d['wsid'])))

        database.discursos.update({'_id': d['_id']}, {'$set': {
            'conteudo_plain_text': plaintext,
            'conteudo_stemmed': stemmed,
        }})


if __name__ == '__main__':
    sys.exit(main(sys.argv))
