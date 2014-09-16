# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import re
import os
import json
import unicodedata
import itertools

from argparse import ArgumentParser

import nltk

import pyth.document
from ofs.local import PTOFS
from pairtree.storage_exceptions import FileNotFoundException
from pyth.plugins.plaintext.writer import PlaintextWriter

# XXX we should use local relative imports, but we are not in a package yet
from rtfreader import CustomRtf15Reader as Rtf15Reader

import clint


def can_be_converted_to_text(p):
    """Return `True` if the given `pyth.document.Paragraph` can be converted
    to text by the PlaintextWriter, `False` otherwise.
    """
    if isinstance(p, pyth.document.Image):
        return False
    return True


def sanitize_rtf_document(doc):
    """Sanitize a `pyth.document.Document`, removing everything that can't be
    converted to plain text.

    WARNING! This method operates in place, changing the input *doc* and
    returning `None`.
    """
    for paragraph in doc.content:
        paragraph.content = filter(can_be_converted_to_text, paragraph.content)


def process_document(doc):

    doc = Rtf15Reader.read(doc)

    # Remove non textual elements from the RTF document
    sanitize_rtf_document(doc)

    # Convert the RTF document to plain text
    doc = PlaintextWriter.write(doc).read().decode('utf-8')

    # Do our best to replace special characters (mostly accentuated chars) with
    # their corresponding transliterated simplified chars
    clean = unicodedata.normalize('NFKD', doc).encode('ascii', 'ignore').decode('utf-8')

    # Do our best to remove punctuation and stuff that don't compose words
    allowed_categories = set(('Lu', 'Ll', 'Nd', 'Zs'))

    filter_function = lambda c: c if unicodedata.category(c) in allowed_categories else '#'

    clean = ''.join(map(filter_function, clean)).replace('#', ' ')

    # We don't want no extra spaces
    clean = re.sub(r'\s+', ' ', clean).lower().strip()

    # XXX Is this really necessary?
    # We only want words
    words = (w for w in clean.split() if not w.isdigit())

    # Reduce words to their stemmed version
    stemmer = nltk.stem.snowball.PortugueseStemmer()

    return ' '.join(itertools.imap(stemmer.stem, words))


def stemmed_bucket(bucket):
    return 'st:' + bucket


def is_stemmed_bucket(bucket):
    return bucket.startswith('st:')


def load_and_prepare_document(storage, bucket, label):
    cache_bucket = stemmed_bucket(bucket)

    try:
        doc = storage.get_stream(cache_bucket, label)
        prep = doc.read()
    except FileNotFoundException:
        doc = storage.get_stream(bucket, label)

        try:
            prep = process_document(doc)
        except Exception:
            print('Failed to load document {0}'.format(label))
            return ''

        # TODO should be like a command line option
        cache_stemmed = True
        if cache_stemmed:
            storage.put_stream(cache_bucket, label, prep)

    return prep


def document_generator(storage, documents_by_author):
    for (author, documents) in documents_by_author.iteritems():

        # Ignore authors which have only one document
        if len(documents) < 2:
            continue

        for bucket, label in documents:
            document = load_and_prepare_document(storage, bucket, label)
            yield (author, document)


def load_documents_from_storage(nbuckets=0):
    # inicializar o armazenamento
    storage = PTOFS()
    storage.list_buckets()

    # All buckets
    buckets = storage.list_buckets()

    # Filter out buckets which are used for caching
    buckets = itertools.ifilterfalse(is_stemmed_bucket, buckets)

    # Filter out all buckets that exceed the limit specified by `args.nbuckets`
    if nbuckets:
        buckets = itertools.islice(buckets, 0, nbuckets)

    # Load documents and authors
    documents_by_author = {}
    document_count = 0

    for bucket in buckets:
        for label in storage.list_labels(bucket):
            author = storage.get_metadata(bucket, label).get('orador')
            document_count += 1
            documents_by_author.setdefault(author, []).append((bucket, label))

    documents = document_generator(storage, documents_by_author)

    return documents, document_count


def main(argv):
    parser = ArgumentParser(prog='stemmer')
    parser.add_argument('--nbuckets', type=int, default=0,
                        help=('the number of buckets you want to process'))
    parser.add_argument('outfile', type=unicode)

    args = parser.parse_args(argv[1:])

    outfile = os.path.expanduser(os.path.expandvars(args.outfile))

    print("Processing documents...")

    documents, document_count = load_documents_from_storage(args.nbuckets)

    documents = clint.textui.progress.bar(documents, expected_size=document_count)

    with open(outfile, 'w') as fp:
        for item in documents:
            fp.write(json.dumps(item) + '\n')

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
