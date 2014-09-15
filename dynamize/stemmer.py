# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import re
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


THIS_DIR = os.path.abspath(os.path.dirname(__file__))


def dtm_as_dataframe(docs, labels=None, **kwargs):
    """Create a DocumentTermMatrix as a pandas DataFrame.

    `**kwargs` will be given directly to `CountVectorizer`.

    *labels* will be used to label the rows
    """

    return df


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


def prepare_document(doc):

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


def all_document_labels(storage, buckets):
    """Return an iterator in the form of [(label, author), ...] of all
    the documents in the given *buckets*
    """
    for bucket in buckets:
        for label in storage.list_labels(bucket):
            author = storage.get_metadata(bucket, label).get('orador')
            yield (label, author)


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
            prep = prepare_document(doc)
        except Exception:
            print('Failed to load document {0}'.format(label))
            return ''

        # TODO should be like a command line option
        cache_stemmed = True
        if cache_stemmed:
            storage.put_stream(cache_bucket, label, prep)

    return prep


def exp_agenda_vonmon(dtm, authors, categories=70, verbose=False, kappa=400):
    """
    * **dtm** must be a pandas.DataFrame
    * **authors** must be a pandas.DataFrame
    """
    rpy2.robjects.r('setwd("{0}")'.format(THIS_DIR))

    rauthors = pandas.rpy.common.convert_to_r_matrix(authors)
    rdtm = pandas.rpy.common.convert_to_r_matrix(dtm)

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

    # chamar o retorica
    result = retorica(rdtm, rauthors)

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


def vonmon_authors_matrix(documents):
    """Filter out authors with less than two documents and generate the
    authors matrix required by vonmon.

    Returns a tuple (authors, labels)
    """
    # Sort by author, convert to list
    documents = sorted(documents, key=lambda d: d[1])

    labels = []
    authors = []
    first_col = 0

    for (author, group) in itertools.groupby(documents, key=lambda x: x[1]):
        group = list(group)

        # Ignore authors with only one document
        if len(group) < 2:
            continue

        last = first_col + len(group) - 1
        first = first_col

        labels.extend([label for (label, _) in group])
        authors.append((first, last))

        first_col = last + 1

    return authors, labels


def main(argv):

    parser = ArgumentParser(prog='stemmer')
    parser.add_argument('--nbuckets', type=int, default=0,
                        help=('the number of buckets you want to use as input'))
    parser.add_argument('--mindf', type=float, default=1.0,
                        help=('When building the vocabulary ignore terms that '
                              'have a document frequency strictly lower than '
                              'the given threshold. This value is also called '
                              'cut-off in the literature. If float, the '
                              'parameter represents a proportion of '
                              'documents, integer absolute counts. Can be'
                              'between 0.0 and 1.0'))

    args = parser.parse_args(argv[1:])

    # inicializar o armazenamento
    storage = PTOFS()
    storage.list_buckets()

    # All buckets
    buckets = storage.list_buckets()

    # Filter out buckets which are used for caching
    buckets = itertools.ifilterfalse(is_stemmed_bucket, buckets)

    # Filter out all buckets that exceed the limit specified by `args.nbuckets`
    if args.nbuckets:
        buckets = itertools.islice(buckets, 0, args.nbuckets)

    # Load labels and authors in the format `iter([(label, author), ...])`
    documents = all_document_labels(storage, buckets)

    # Filter out invalid authors and generate the Authors Matrix required by vonmon
    authors, labels = vonmon_authors_matrix(documents)

    # Load and prepare the documents
    def _load_and_prepare_document(label):
        bucket = label.split(':')[0]
        return load_and_prepare_document(storage, bucket, label)

    print('Loading documents...')

    # Generate the Document Term Matrix
    corpus = itertools.imap(_load_and_prepare_document, labels)

    mindf = max(1.0, args.mindf)
    mindf = min(0.0, mindf)

    cv = CountVectorizer(max_df=mindf)
    ft = cv.fit_transform(corpus)

    import pdb; pdb.set_trace()

    # XXX FIXME Requires too much memory for a 81k x 74k dtm
    dtm = pandas.DataFrame(ft.toarray(), index=labels,
                           columns=cv.get_feature_names())

    # R matrices are 1-indexed, not 0-indexed, and this is an awesome trick :)
    authors = pandas.DataFrame(numpy.matrix(authors))
    authors = authors.radd(1)

    print('Aplicando vonmon a {0} documentos, {1} palavras e {2} autores...'.format(
        len(dtm.index), len(dtm.columns), len(authors)
    ))

    return exp_agenda_vonmon(dtm, authors)

if __name__ == '__main__':
    import sys
    main(sys.argv)
