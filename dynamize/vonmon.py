# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import re
import sys
import json
import os.path
import argparse
import datetime
import itertools
from operator import itemgetter

import errno
import numpy
import pandas
import pandas.rpy
import pymongo
import rpy2.robjects
from sklearn.feature_extraction.text import CountVectorizer
from clint.textui import puts, progress
from .utils.transliterate import transliterate

from utils.stem import stem_rtf_file

import os.path


def here(*args):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), *args))


def merge(*dicts):
    r = dict()
    map(r.update, reversed(dicts))
    return r


# XXX BEHOLD DA' MONKEYPATCH BELOW!!!
#
# For whatever reason, `pandas.rpy.common` gives error when looking up for
# ordinary numpy types from its `VECTOR_TYPES` dict. Like it's looking up for
# `numpy.int64`, which is in the dict, but it's another instance of
# `numpy.int64`, so it's `id()` is different. Since I don't know why that is
# happening nor how to prevent it, I'll simply monkeypatch that dict and make
# it lookup types by name. It works for us.


class LookupByNameTypeDict(dict):
    def __init__(self, actual):
        super(LookupByNameTypeDict, self).__init__((str(t), v) for (t, v) in actual.items())

    def __getitem__(self, key):
        return super(LookupByNameTypeDict, self).__getitem__(str(key))

pandas.rpy.common.VECTOR_TYPES = LookupByNameTypeDict(pandas.rpy.common.VECTOR_TYPES)

# END OF MONKEYPATCH: We're *safe* now.


def exp_agenda_vonmon(dtm, authors, ncats=70, verbose=False, kappa=400):
    """
    Call `exp.agenda.vonmon` through `rpy2`'s R interface.

    :param dtm: the Document Term Matrix
    :type dtm: pandas.DataFrame
    :param authors: the Authors matrix
    :type authors: pandas.DataFrame
    """
    THIS_DIR = os.path.abspath(os.path.dirname(__file__))

    # Now we prepare our data for `exp.agenda.vonmon`
    # 1) Convert the DTM to a R matrix
    # 2) Convert the authors matrix to a R matrix and sum it with 1 (because
    #    exp.agenda.vonmon uses 1-indexing, while the matrix is currently
    #    0-indexed)

    dtm = pandas.rpy.common.convert_to_r_matrix(dtm)
    authors = pandas.rpy.common.convert_to_r_matrix(authors.radd(1))

    rpy2.robjects.r('setwd("{0}")'.format(THIS_DIR))

    # Load `exp.agenda.vonmon` into the rpy2 environment
    rpy2.robjects.r("source('../r/ExpAgendVMVA.R')")

    exp_agenda_vonmon = rpy2.robjects.r('exp.agenda.vonmon')

    # Call it
    topics = exp_agenda_vonmon(term_doc=dtm, authors=authors, n_cats=ncats,
                               verbose=verbose, kappa=kappa)

    puts('Saving results...')

    rsave_rds = rpy2.robjects.r('saveRDS')
    rsave_rds(topics, file='topics_{0}.Rda'.format(shortnow()))

    return pandas.rpy.common.convert_robj(topics)


def build_authors_matrix(authors):
    start = 0

    names = []
    matrix = []

    authors = list(authors)

    while authors:
        cur = authors[0]
        count = authors.count(cur)

        matrix.append((start, start + count - 1))

        names.append(cur)

        start += count
        authors = authors[count:]

    return pandas.DataFrame(matrix, index=names)


class DocumentsFile(object):
    @classmethod
    def open(cls, filename):
        return cls(filename)

    def __init__(self, filename):
        self._fp = open(filename, 'r')
        self._line_count = sum(1 for _ in self._fp)
        self._fp.seek(0)

    def __len__(self):
        return self._line_count

    def __iter__(self):
        for line in self._fp:
            yield json.loads(line)

    def close(self):
        self._fp.close()


class Collector(list):
    def __init__(self, cb):
        self._callback = cb
        super(Collector, self).__init__()

    def collect(self, item):
        self.append(item)
        return self._callback(item)


def eliminate_authors_with_only_one_speech(docs, func):
    cur = None
    prevdoc = None
    doccount = False

    for doc in docs:
        author = func(doc)
        if author != cur:
            cur = author
            prevdoc = doc
            doccount = False
        elif not doccount:
            yield prevdoc
            yield doc
            doccount = True
        else:
            yield doc


def mkdirp(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if not (exc.errno == errno.EEXIST and os.path.isdir(path)):
            raise


def shortnow(now=None):
    if now is None:
        now = datetime.datetime.now()
    return now.strftime(r'%Y%m%d%H%M%S')


def mkdate(s):
    try:
        return datetime.datetime.strptime(s, '%Y-%m-%d')
    except Exception, e:
        puts("Could not parse date `{0}'. Apparently not in the format Y-m-d: {1}".format(s, e))


class WriteableDir(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        prospective_dir = values

        try:
            mkdirp(prospective_dir)
        except OSError as e:
            raise argparse.ArgumentError(prospective_dir, e.message)


def normalize_name(name):
    # strip AKIRA OTSUBO (PRESIDENTE)
    name = re.sub(r'\s*\([^\)]+\)\s*$', '', name)

    # strip ALGUEM, ALGUMA COISA
    name = re.sub(r'\s*,.*$', '', name)

    # strip ALGUEM - PARLAMENTAR JOVEM
    # but don't touch AKIRA-TO
    name = re.sub(r'\s+-.*$', '', name)

    return name


def main(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('-H', '--host', type=unicode, default='localhost')
    parser.add_argument('-P', '--port', type=int, default=27017)
    parser.add_argument('-d', '--database', type=unicode, default='retorica_development')

    parser.add_argument('--mindf', type=float, default=1.0,
                        help='Minimum document frequency for cuts')
    parser.add_argument('--maxdf', type=float, default=1.0,
                        help='Maximum document frequency for cuts')
    parser.add_argument('--ncats', type=int, default=70,
                        help='Number of categories in the generated matrix')
    parser.add_argument('-o', '--output_directory', action=WriteableDir,
                        default=None, help=('Directory where the output data will be saved. '
                                            'If it doesnt exist well create it. '
                                            'Defaults to ./vonmon/<rightnow>.'))

    parser.add_argument('--initial-term', type=mkdate, default=None,
                        help=('ignore all documents previously to this date (in the Y-m-d format)'))
    parser.add_argument('--final-term', type=mkdate, default=None,
                        help=('ignore all documents after this date (in the Y-m-d format)'))
    parser.add_argument('--phases', type=unicode, default='*',
                        help=('Fases da sessão, separadas por vírgula: '
                              'PE,BC,AB,OD,HO,CG,GE. * para todos.'))

    args = parser.parse_args(argv[1:])

    AUTHOR_COLUMN = 'vonmon_author'
    CONTENT_COLUMN = 'vonmon_content'

    # Sanitize arguments
    args.mindf = max(min(args.mindf, 1.0), 0.0)
    args.maxdf = max(min(args.maxdf, 1.0), 0.0)

    if args.maxdf == 1.0:
        args.maxdf = 1

    # Create dirs right now to avoid problems later!
    output_folder = args.output_directory
    if not output_folder:
        output_folder = os.path.abspath(os.path.join(
            os.getcwd(), 'vonmon',
            datetime.datetime.now().strftime('%Y-%m-%d_%H%M')))

    # Make sure it exists!
    mkdirp(output_folder)

    def outputpath(*args):
        return os.path.abspath(os.path.join(output_folder, *args))

    # Connect to the database
    client = pymongo.MongoClient(args.host, args.port)
    database = getattr(client, args.database)

    # Lookup documents
    lookup = dict()

    if args.phases and args.phases != '*':
        phases = map(lambda s: s.strip(), args.phases.split(','))
        lookup['faseSessao.codigo'] = {'$in': phases}

    dt_lookup = {}

    if args.initial_term:
        dt_lookup['$gte'] = args.initial_term
    if args.final_term:
        dt_lookup['$lte'] = args.final_term

    if dt_lookup:
        lookup['horaInicioDiscurso'] = dt_lookup

    puts(repr(lookup))

    collection = database.discursos

    puts("Normalizing author names...")

    speeches = collection.find(merge(lookup, {AUTHOR_COLUMN: {'$exists': False}}))

    for s in progress.bar(speeches, expected_size=speeches.count()):
        collection.update({'_id': s['_id']}, {
            '$set': {
                AUTHOR_COLUMN: transliterate(normalize_name(s['orador']['nome'])),
            },
        })

    lookup = merge(lookup, {AUTHOR_COLUMN: {'$exists': True}})

    puts("Preparing the corpus...")

    speeches = collection.find(lookup)

    for s in progress.bar(speeches, expected_size=speeches.count()):
        f = s['files']

        if len(f) != 1:
            puts("Ignoring {s}: it has {n} files!".format(s=s['_id'], n=len(f)))
            continue

        f = f[0]

        with open(here('kingsnake', f['path'])) as rtf_file:
            content = stem_rtf_file(rtf_file)

        collection.update({'_id': s['_id']}, {
            '$set': {
                CONTENT_COLUMN: content,
            },
        })

    lookup = merge(lookup, {CONTENT_COLUMN: {'$exists': True}})

    puts("Optimizing the corpus...")

    collection.ensure_index(AUTHOR_COLUMN)

    speeches = collection.find(lookup).sort(AUTHOR_COLUMN)

    speeches = progress.bar(speeches, expected_size=speeches.count())

    speeches = eliminate_authors_with_only_one_speech(speeches, itemgetter(AUTHOR_COLUMN))

    collector = Collector(itemgetter(CONTENT_COLUMN))

    corpus = itertools.imap(collector.collect, speeches)

    # Generate the Document Term Matrix

    puts("Building the DTM...")

    cv = CountVectorizer(min_df=args.mindf, max_df=args.maxdf)
    dtm = cv.fit_transform(corpus)

    documents_and_authors = pandas.DataFrame([
        (d['_id'], d[AUTHOR_COLUMN]) for d in collector
    ])
    documents_and_authors.to_csv(outputpath('documents_and_authors.csv'), index=False, header=False, encoding='utf-8')

    authors = build_authors_matrix(documents_and_authors.icol(1).as_matrix())

    authors.to_csv(outputpath('authors.csv'), header=False, encoding='utf-8')

    puts("")

    puts("DTM has {0} documents and {1} terms:".format(
        dtm.shape[0], dtm.shape[1],
    ))

    puts("")

    dtm = pandas.DataFrame(dtm.toarray(), columns=cv.get_feature_names())

    puts("Calling exp.agenda.vonmon through rpy2...")

    topics = exp_agenda_vonmon(dtm, authors, ncats=args.ncats, verbose=True)

    topics['thetas'].index = authors.index

    for key in topics.keys():
        filename = outputpath('{0}.csv'.format(key))
        topics[key].to_csv(filename, encoding='utf-8')

    thetas = topics['thetas']
    author_topics = thetas.idxmax(axis=1)

    # Create a proportion table with the results
    t = []
    for i in range(thetas.shape[0]):
        row = thetas.irow(i)
        t.append(row / numpy.sum(row))

    propthetas = pandas.DataFrame(t, index=thetas.index)

    result = []
    for i, topic_idx in enumerate(propthetas.idxmax(axis=1)):
        row = propthetas.irow(i)
        enfase = row[topic_idx]
        result.append((topic_idx, enfase))

    result = pandas.DataFrame(result, index=propthetas.index)
    result.to_csv(outputpath('result.csv'), encoding='utf-8')

    mus = topics['mus']

    topicwords = []
    for i in range(mus.shape[1]):
        words = mus.icol(i).sort(ascending=False, inplace=False)
        topicwords.append([w for (w, e) in words[:30].iteritems()])

    words = pandas.DataFrame(topicwords)
    words.to_csv(outputpath('words.csv'), header=False, index=False, encoding='utf-8')


if __name__ == '__main__':
    sys.exit(main(sys.argv) or 0)
