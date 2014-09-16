# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import re
import json
import unicodedata
import itertools
import os.path

from argparse import ArgumentParser

import nltk
import numpy
import pandas
import pandas.rpy.common
import rpy2.robjects

import pyth.document
from ofs.local import PTOFS
from pairtree.storage_exceptions import FileNotFoundException
from pyth.plugins.plaintext.writer import PlaintextWriter
from sklearn.feature_extraction.text import CountVectorizer

# XXX we should use local relative imports, but we are not in a package yet
from rtfreader import CustomRtf15Reader as Rtf15Reader

import clint


THIS_DIR = os.path.abspath(os.path.dirname(__file__))


def exp_agenda_vonmon(dtm, authors, categories=70, verbose=False, kappa=400):
    """
    * **dtm** must be a pandas.DataFrame
    * **authors** must be a pandas.DataFrame
    """
    rpy2.robjects.r('setwd("{0}")'.format(THIS_DIR))

    retorica = r'''
    retorica <- function(dtm, autorMatrix, ncats=70, verbose=T, kappa=400) {

    topics <- exp.agenda.vonmon(term.doc = dtm, authors = autorMatrix,
                                n.cats = ncats, verbose = verbose, kappa = kappa)

    # Definindo topicos de cada autor e arquivo final
    autorTopicOne <- NULL
    for( i in 1:dim(topics[[1]])[1]){
    autorTopicOne[i] <- which.max(topics[[1]][i,])
    }

    # compute the proportion of documents from each author to each topic
    autorTopicPerc <- prop.table(topics[[1]], 1)

    autorTopicOne <- as.data.frame(autorTopicOne)

    for( i in 1:nrow(autorTopicOne)){
    autorTopicOne$enfase[i] <- autorTopicPerc[i,which.max(autorTopicPerc[i,])]
    }

    topics$one <- autorTopicOne

    save("topics", file="topics.RData");

    return(topics)
    }
    '''

    # carregar o vonmon
    rpy2.robjects.r("source('../r/ExpAgendVMVA.R')")

    # carregar o retorica
    retorica = rpy2.robjects.r(retorica)

    rdtm = pandas.rpy.common.convert_to_r_matrix(dtm)
    rauthors = pandas.rpy.common.convert_to_r_matrix(authors)

    # chamar o retorica
    result = retorica(rdtm, rauthors, categories, verbose, kappa)

    print('Salvando resultados...')
    print('topics.csv...')

    # temas relevantes estão salvos na variável `topics$one`
    topics = pandas.rpy.common.convert_robj(result[4])

    topics.index = author_names
    topics.columns = ('tema', 'enfase')

    topics.to_csv(os.path.join(THIS_DIR, 'topics.csv'), encoding='utf-8')

    print('topic_words.csv...')

    write_table = rpy2.robjects.r('write.table')
    write_table(result[1], file='topic_words.csv', sep=',', row_names=True)

    print('Feito!')


def build_authors_matrix(authors):
    start = 0

    matrix = []

    while authors:
        cur = authors[0]
        count = authors.count(cur)

        matrix.append((start, start + count - 1))

        start += count
        authors = authors[count:]

    # R matrices are 1-indexed, not 0-indexed, and this is an awesome trick :)
    matrix = pandas.DataFrame(numpy.matrix(matrix))
    return matrix.radd(1)


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
        item, collected = self._callback(item)
        self.append(collected)
        return item


def main(argv):
    parser = ArgumentParser(prog='stemmer')
    parser.add_argument('--mindf', type=float, default=1.0,
                        help='Minimum document frequency for cuts')
    parser.add_argument('--maxdf', type=float, default=1.0,
                        help='Maximum document frequency for cuts')

    parser.add_argument('docsfile', type=unicode)

    args = parser.parse_args(argv[1:])

    documents = DocumentsFile.open(args.docsfile)
    document_count = len(documents)

    documents = clint.textui.progress.bar(documents)

    def collect_author(item):
        author, document = item
        return document, author

    authors = Collector(collect_author)

    corpus = itertools.imap(authors.collect, documents)

    # Generate the Document Term Matrix

    clint.textui.puts("Building the DTM...")

    mindf = max(min(args.mindf, 1.0), 0.0)
    maxdf = max(min(args.maxdf, 1.0), 0.0)

    if maxdf == 1.0:
        maxdf = 1

    cv = CountVectorizer(min_df=mindf, max_df=maxdf)
    ft = cv.fit_transform(corpus)

    authors = build_authors_matrix(authors)

    clint.textui.puts(
        'Aplicando vonmon a {0} documentos, {1} termos e {2} autores...'.format(
            document_count, len(cv.vocabulary_), len(authors),
    ))


    # XXX FIXME Requires too much memory for a 81k x 74k DTM
    clint.textui.puts("Transforming DTM...")

    dtm = pandas.DataFrame(ft.toarray(), columns=cv.get_feature_names())

    return exp_agenda_vonmon(dtm, authors)

if __name__ == '__main__':
    import sys
    main(sys.argv)
