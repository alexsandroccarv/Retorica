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
from copy import copy

import nltk
import pandas
import pandas.rpy
import rpy2.robjects
import pymongo
import errno
from clint.textui import indent, puts, progress, colored
from clint.textui import progress
from clint.textui import puts, colored

from scrapy.command import ScrapyCommand
from scrapy.exceptions import UsageError
from scrapy.utils.conf import arglist_to_dict
from sklearn.feature_extraction.text import CountVectorizer

from kingsnake.utils import speech_collection_from_command


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

def curry_getattr(key, default=None):
    return lambda x: getattr(x, key, default)


def merge(*d):
    """Merge a bunch of dicts.
    """
    r = {}
    for i in reversed(d):
        r.update(i)
    return r


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


def ignore_authors_without_enough_speeches(docs, func):
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

def dir_exists_and_is_writable(path):
    # TODO actually implement this
    return True


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



def check_date(option, opt, value):
    try:
        return datetime.datetime.strptime(value, '%Y-%m-%d')
    except Exception, e:
        raise OptionValueError(
            ("Could not parse date `{0}'. "
             "Are you sure it's in the `Y-m-d' format?").format(value))


def decorate_option_class_with_date_checker(Option):

    class CustomOption(Option):
        TYPES = Option.TYPES + ('date',)
        TYPE_CHECKER = copy(Option.TYPE_CHECKER)
        TYPE_CHECKER['date'] = check_date

    return CustomOption



class Command(ScrapyCommand):

    requires_project = True

    def syntax(self):
        return "[options] [output_directory]"

    def short_desc(self):
        return "Do everything that needs to be done"

    def add_options(self, parser):
        ScrapyCommand.add_options(self, parser)

        parser.option_class = \
            decorate_option_class_with_date_checker(parser.option_class)

        parser.add_option('--categories', type=int, default=70,
                          help='Number of categories in the generated matrix')

        parser.add_option('--mindf', type=float, default=0.0,
                            help='Minimum document frequency for cuts')
        parser.add_option('--maxdf', type=float, default=1.0,
                            help='Maximum document frequency for cuts')

        parser.add_option('--initial-term', type='date', default=None,
                            help=('Ignore documents prior to this date (use Y-m-d format)'))
        parser.add_option('--final-term', type='date', default=None,
                            help=('Ignore documents after this date (use Y-m-d format)'))

        parser.add_option('--phases', default='*',
                          help=('Fases da sessão, separadas por vírgula: '
                                'PE,BC,AB,OD,HO,CG,GE. * para todos.'))

        parser.add_option('-o', '--output_directory',
                            default=None, help=('Directory where the output data will be saved. '
                                                'If it doesnt exist we\'ll create it. '
                                                'Defaults to ./vonmon/<A_TIMESTAMP>.'))

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
        else:
            opts.phases = None

        if not opts.output_directory:
            fname = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M')
            relpath = os.path.join(os.getcwd(), 'vonmon', fname)
            opts.output_directory = os.path.abspath(relpath)

        # Make sure the output directory exists and is writeable before running
        # anything
        # TODO dump params, info, etc into the directory
        mkdirp(opts.output_directory)

        if not dir_exists_and_is_writable(opts.output_directory):
            raise RuntimeError('The selected output directory is not writeable')

        # Save it in the settings so we can use it through the command
        self.settings.set('VONMON_OPTIONS', opts)
        self.settings.set('VONMON_OUTPUT_DIRECTORY', opts.output_directory)

    def collection(self):
        if not hasattr(self, '_speech_collection'):
            self._speech_collection = speech_collection_from_command(self)
        return self._speech_collection

    def run(self, args, opts):
        lookup = {}

        if opts.phases:
            lookup['faseSessao.codigo'] = {'$in': opts.phases}

        dt_lookup = {}

        if opts.initial_term:
            dt_lookup['$gte'] = opts.initial_term

        if opts.final_term:
            dt_lookup['$lte'] = opts.final_term

        if dt_lookup:
            lookup['horaInicioDiscurso'] = dt_lookup

        # Prepare *ALL THE UNPREPARED* documents!!!
        l = merge(lookup, {
            'vonmon': {'$exists': False}
        })
        print(l)
        documents = self.collection().find(l)

        for d in progress.bar(documents, expected_size=documents.count()):

            self.collection().update({'_id': d.get('_id')}, {
                '$set': {
                    'vonmon': {
                        'author': author,
                        'stemmed': stemmed,
                        'text': text,
                    },
                },
            })

        # CALL THE ALGO!!!
        lookup.update({
            'vonmon': {'$exists': True}
        })

        self.collection().ensure_index('author')

        documents = self.collection().find(lookup).sort('author')

        documents = ignore_authors_without_enough_speeches(documents, curry_getattr('author'))

        collector = Collector(lambda i: i.get('text'))

        corpus = itertools.imap(collector.collect, documents)

        # Generate a Document Term Matrix
        cv = CountVectorizer(min_df=opts.mindf, max_df=opts.maxdf)
        dtm = cv.fit_transform(corpus)

        document_authors_matrix = pandas.DataFrame([
            (d.get('author'), d.get('_id')) for d in collector
        ])

        # Output the data
        document_authors_matrix.to_csv(self.outputpath('authors.csv'),
                                       index=False, header=False, encoding='utf-8')

        authors = build_authors_matrix(document_authors_matrix.icol(1).as_matrix())

        puts("DTM has {0} documents and {1} terms:".format(
            dtm.shape[0], dtm.shape[1],
        ))

        puts("")

        dtm = pandas.DataFrame(dtm.toarray(), columns=cv.get_feature_names())

        puts("Calling exp.agenda.vonmon through rpy2...")

        topics = exp_agenda_vonmon(dtm, authors, ncats=opts.categories, verbose=True)

        # Index result lines with author names for readability
        topics['thetas'].index = authors.index

        for key in topics.keys():
            filename = self.outputpath('{-1}.csv'.format(key))
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
        result.to_csv(self.outputpath('result.csv'), encoding='utf-8')

        mus = topics['mus']

        topicwords = []
        for i in range(mus.shape[1]):
            words = mus.icol(i).sort(ascending=False, inplace=False)
            topicwords.append([w for (w, e) in words[:30].iteritems()])

        words = pandas.DataFrame(topicwords)
        words.to_csv(outputpath('words.csv'), header=False, index=False, encoding='utf-8')
