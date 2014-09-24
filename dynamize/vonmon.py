# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import json
import itertools
import os.path
from argparse import ArgumentParser

import clint
import numpy
import pandas
import pandas.rpy
import rpy2.robjects
from sklearn.feature_extraction.text import CountVectorizer

from rinterface import convert_to_r_matrix


THIS_DIR = os.path.abspath(os.path.dirname(__file__))


def exp_agenda_vonmon(dtm, authors, categories=70, verbose=False, kappa=400):
    """
    Call `exp.agenda.vonmon` through `rpy2`'s R interface.

    :param dtm: the Document Term Matrix
    :type dtm: rpy2.robjects.vectors.Matrix
    :param dtm: the Authors matrix
    :type dtm: rpy2.robjects.vectors.Matrix
    """
    rpy2.robjects.r('setwd("{0}")'.format(THIS_DIR))

    retorica = r'''
    retorica <- function(dtm, autorMatrix, ncats=70, verbose=T, kappa=400) {

    save("dtm", file="dtm.RData")

    topics <- exp.agenda.vonmon(term.doc = dtm, authors = autorMatrix,
                                n.cats = ncats, verbose = verbose, kappa = kappa)

    # Definindo topicos de cada autor e arquivo final
    autorTopicOne <- NULL
    for (i in 1:dim(topics[[1]])[1]) {
        autorTopicOne[i] <- which.max(topics[[1]][i,])
    }

    # compute the proportion of documents from each author to each topic
    autorTopicPerc <- prop.table(topics[[1]], 1)

    autorTopicOne <- as.data.frame(autorTopicOne)

    for (i in 1:nrow(autorTopicOne)) {
        autorTopicOne$enfase[i] <- autorTopicPerc[i,which.max(autorTopicPerc[i,])]
    }

    topics$one <- autorTopicOne

    save("topics", file="topics.RData");

    return(topics)
    }
    '''

    # load `exp.agenda.vonmon` into the rpy2 environment
    rpy2.robjects.r("source('../r/ExpAgendVMVA.R')")

    # load our glue code into the environment
    retorica = rpy2.robjects.r(retorica)

    # call it!
    result = retorica(dtm, authors, categories, verbose, kappa)

    clint.textui.puts('Salvando resultados...')

    with clint.textui.indent(2):
        # XXX FIXME We should probably do this in Python
        clint.textui.puts('topic_words.csv...')

        write_table = rpy2.robjects.r('write.table')
        write_table(result[1], file='topic_words.csv', sep=',', row_names=True)

        clint.textui.puts('topics.csv...')

        # temas relevantes estão salvos na variável `topics$one`
        #topics = pandas.rpy.common.convert_robj(result[4])
        write_table(result[4], file='topics.csv', sep=',', row_names=True)

        # XXX FIXME We should really find a way to fill this in with
        # the names of the author
        #topics.index = author_names
        #topics.columns = ('tema', 'enfase')

        #topics.to_csv(os.path.join(THIS_DIR, 'topics.csv'), encoding='utf-8')

    return result


def build_authors_matrix(authors):
    start = 0

    names = []
    matrix = []

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
        item, collected = self._callback(item)
        self.append(collected)
        return item


class ProxiedBar(object):
    def __init__(self, it, *args, **kwargs):
        self._it = it
        self._bar = clint.textui.progress.bar(self._it, *args, **kwargs)

    def __iter__(self):
        for i in self._bar:
            yield i

    def __getattr__(self, name):
        return getattr(self._it, name)


def main(argv):
    parser = ArgumentParser(prog='stemmer')
    parser.add_argument('--mindf', type=float, default=1.0,
                        help='Minimum document frequency for cuts')
    parser.add_argument('--maxdf', type=float, default=1.0,
                        help='Maximum document frequency for cuts')

    parser.add_argument('docsfile', type=unicode)

    args = parser.parse_args(argv[1:])

    # Sanitize arguments
    args.mindf = max(min(args.mindf, 1.0), 0.0)
    args.maxdf = max(min(args.maxdf, 1.0), 0.0)

    if args.maxdf == 1.0:
        args.maxdf = 1

    documents = DocumentsFile.open(args.docsfile)

    documents = clint.textui.progress.bar(documents)

    def collect_author(item):
        author, document = item
        return document, author

    authors = Collector(collect_author)

    corpus = itertools.imap(authors.collect, documents)

    # Generate the Document Term Matrix
    clint.textui.puts("Building the DTM...")

    cv = CountVectorizer(min_df=args.mindf, max_df=args.maxdf)
    dtm = cv.fit_transform(corpus)

    # Dense is faster!
    #dtm = dtm.todense()
    dtm = pandas.DataFrame(dtm.toarray(), columns=cv.get_feature_names())

    authors = build_authors_matrix(authors)

    clint.textui.puts("DTM has {0} documents and {1} terms:".format(
        dtm.shape[0], dtm.shape[1],
    ))
    
    clint.textui.puts("")
    
    #wordfreq = pd.DataFrame({'topics': cv.get_feature_names(), 'occurrences': numpy.asarray(dtm.sum(axis=0)).ravel()})
    #wordfreq.sort('occurrences', ascending=False, inplace=True)

    #print(wordfreq)
    
    #clint.textui.puts("")
    #clint.textui.puts('')
    #import sys; sys.exit(1)

    # Sum 1, as explained above
    #for a in authors.index:
    #    print(a.encode('utf-8'))

    authors = pandas.rpy.common.convert_to_r_matrix(authors.radd(1))

    # Now we prepare our data for `exp.agenda.vonmon`
    # 1) Convert the DTM to a R matrix
    # 2) Convert the authors matrix to a R matrix and sum it with 1 (because
    #    exp.agenda.vonmon uses 1-indexing, while the matrix is currently
    #    0-indexed)

    clint.textui.puts("Preparing the DTM...")

    #dtm = ProxiedBar(dtm, expected_size=dtm.shape[0])

    class MyDict(dict):
        def __init__(self, actual):
            super(MyDict, self).__init__((str(t), v) for (t, v) in actual.items())

        def __getitem__(self, key):
            return super(MyDict, self).__getitem__(str(key))

    pandas.rpy.common.VECTOR_TYPES = MyDict(pandas.rpy.common.VECTOR_TYPES)

    dtm = pandas.rpy.common.convert_to_r_matrix(dtm)

    dtm.colnames = rpy2.robjects.StrVector(cv.get_feature_names())

    clint.textui.puts("Calling exp.agenda.vonmon through rpy2...")

    return exp_agenda_vonmon(dtm, authors, verbose=True)


if __name__ == '__main__':
    import sys
    main(sys.argv)
