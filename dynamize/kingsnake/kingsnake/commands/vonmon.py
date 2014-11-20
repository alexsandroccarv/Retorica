# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import argparse
import datetime
import itertools
import json
import numpy
import os
import os.path
import pymongo
import re
import sys
import unicodedata
from StringIO import StringIO
from argparse import ArgumentParser

import nltk
import pyth.document
import pandas
import pandas.rpy
import rpy2.robjects
import pymongo
import errno
from clint.textui import indent, puts, progress, colored
from clint.textui import progress
from clint.textui import puts, colored
from pyth.plugins.plaintext.writer import PlaintextWriter
from scrapy.command import ScrapyCommand
from scrapy.exceptions import UsageError
from scrapy.utils.conf import arglist_to_dict
from sklearn.feature_extraction.text import CountVectorizer

from kingsnake.pipelines import DiscursosMongoDBPipeline
from kingsnake.utils.rtfreader import CustomRtf15Reader as Rtf15Reader


# XXX BEHOLD DA' MONKEYPATCH BELOW!!!
#
# For whatever reason, `pandas.rpy.common` gives error when looking up for
# ordinary numpy types from its `VECTOR_TYPES` dict. Like it's looking up for
# `numpy.int64`, which is in the dict, but it's another instance of
# `numpy.int64`, so it's `id()` is different. Since I don't know why this is
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



class Command(ScrapyCommand):

    requires_project = True

    def syntax(self):
        return "[options] [output_directory]"

    def short_desc(self):
        return "Do everything that needs to be done"

    def add_options(self, parser):
        ScrapyCommand.add_options(self, parser)

        parser.add_option('--mindf', type=float, default=1.0,
                            help='Minimum document frequency for cuts')
        parser.add_option('--maxdf', type=float, default=1.0,
                            help='Maximum document frequency for cuts')
        parser.add_option('--ncats', type=int, default=70,
                            help='Number of categories in the generated matrix')

        parser.add_option('--initial-term', type=mkdate, default=None,
                            help=('ignore all documents previously to this date (in the Y-m-d format)'))
        parser.add_option('--final-term', type=mkdate, default=None,
                            help=('ignore all documents after this date (in the Y-m-d format)'))
        parser.add_option('--phases', type=unicode, default='*',
                            help=('Fases da sessão, separadas por vírgula: '
                                'PE,BC,AB,OD,HO,CG,GE. * para todos.'))

        parser.add_option('-o', '--output_directory',
                            default=None, help=('Directory where the output data will be saved. '
                                                'If it doesnt exist well create it. '
                                                'Defaults to ./vonmon/<rightnow>.'))

    def outputpath(self, *args):
        base = self.settings.get('VONMON_OUTPUT_DIRECTORY')
        return os.path.abspath(os.path.join(base, *args))

    def process_options(self, args, opts):
        ScrapyCommand.process_options(self, args, opts)

        opts.mindf = max(min(opts.mindf, 1.0), 0.0)
        opts.maxdf = max(min(opts.maxdf, 1.0), 0.0)

        if opts.maxdf == 1.0:
            opts.maxdf = 1

        if opts.phases and opts.phases != '*':
            opts.phases = [p.strip() for p in opts.phases.split(',')]

        if not opts.output_directory:
            fname = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M')
            relpath = os.path.join(os.getcwd(), 'vonmon', fname)
            opts.output_directory = os.path.abspath(relpath)

        # Make sure the output directory exists and is writeable
        # before running anything
        # TODO make sure its writeable
        # TODO dump params, info, etc into the directory
        mkdirp(opts.output_directory)

        # Save it in the settings so we can use it through the command
        self.settings.set('VONMON_OUTPUT_DIRECTORY', opts.output_directory)

    def run(self, args, opts):
        # Lookup documents
        lookup = {
            'conteudo_stemmed': {'$exists': True},
            '$and': [
                {'deputy_id': {'$exists': True}},
                {'deputy_id': {'$ne': None}},
            ],
        }

        if opts.phases:
            lookup['fase_sessao.codigo'] = {'$in': opts.phases}

        dt_lookup = {}

        if opts.initial_term:
            dt_lookup['$gte'] = opts.initial_term

        if opts.final_term:
            dt_lookup['$lte'] = opts.final_term

        if dt_lookup:
            lookup['proferido_em'] = dt_lookup

        # Convert documents to text
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

        AUTHOR_COLUMN = 'author_clean'
        CONTENT_COLUMN = 'conteudo_stemmed'

        author_column = 'author_clean'

        database.discursos.ensure_index(author_column)

        documents = database.discursos.find(lookup).sort(author_column)

        documents = progress.bar(documents, expected_size=documents.count())

        documents = eliminate_authors_with_only_one_speech(documents, lambda x: x[AUTHOR_COLUMN])

        collector = Collector(lambda i: i.get(CONTENT_COLUMN))

        corpus = itertools.imap(collector.collect, documents)

        # Generate the Document Term Matrix
        puts("Building the DTM...")

        cv = CountVectorizer(min_df=args.mindf, max_df=args.maxdf)
        dtm = cv.fit_transform(corpus)

        document_authors_matrix = pandas.DataFrame([
            (d[AUTHOR_COLUMN], d['_id']) for d in collector
        ])

        document_authors_matrix.to_csv(self.outputpath('authors.csv'),
                                       index=False, header=False, encoding='utf-8')

        authors = build_authors_matrix(document_authors_matrix.icol(1).as_matrix())

        # XXX FIXME Is this really relevant?
        authors.to_csv(outputpath('authors_old.csv'), header=False, encoding='utf-8')

        puts("")

        puts("DTM has {0} documents and {1} terms:".format(
            dtm.shape[0], dtm.shape[1],
        ))

        puts("")

        dtm = pandas.DataFrame(dtm.toarray(), columns=cv.get_feature_names())

        puts("Calling exp.agenda.vonmon through rpy2...")

        topics = exp_agenda_vonmon(dtm, authors, ncats=args.ncats, verbose=True)

        # Index result lines with author names
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
