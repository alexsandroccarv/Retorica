# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import re
import unicodedata
import itertools

import nltk
import numpy
from ofs.local import PTOFS
from pyth.plugins.plaintext.writer import PlaintextWriter
from pyth.plugins.rtf15.reader import Rtf15Reader
import pandas.rpy.common
import rpy2.robjects
from rpy2.robjects.packages import importr
import os.path
import pandas
from sklearn.feature_extraction.text import CountVectorizer


IGNORE_WORDS = set([
    # preposições
    'por', 'para', 'a', 'ante', 'ate', 'apos', 'de', 'desde', 'em', 'entre',
    'com', 'sem', 'sob', 'sobre',

    # artigos e combinações
    'a', 'o', 'ao', 'da', 'do', 'pelo', 'pela'

    # pronomes
    'eu', 'tu', 'ele', 'ela', 'nós', 'vós', 'eles', 'elas', 'voce', 'me',
    'mim', 'se',

    # etc
    'que', 'sr', 'sra'
])

IGNORE_STEMS = set([
    'porq', 'nao', 'sao', 'quer', 'ser'
])


def dtm_as_dataframe(docs, labels=None, **kwargs):
    """Create a DocumentTermMatrix as a pandas DataFrame.

    `**kwargs` will be given directly to `CountVectorizer`.

    *labels* will be used to label the rows
    """
    vectorizer = CountVectorizer(**kwargs)
    x1 = vectorizer.fit_transform(docs)
    df = pandas.DataFrame(x1.toarray(), columns=vectorizer.get_feature_names())
    if labels:
        df.index = labels
    return df

def prepare_document(doc):
    # Converter o documento do formato RTF para plaintext
    doc = Rtf15Reader.read(doc)
    doc = PlaintextWriter.write(doc).read().decode('utf-8')

    # Remover caracteres especiais e acentuação
    clean = unicodedata.normalize('NFKD', doc).encode('ascii', 'ignore').decode('utf-8')

    # Remover pontuação e outros caracteres que não compõem palavras
    allowed_categories = set(('Lu', 'Ll', 'Nd', 'Zs'))
    filter_function = lambda c: c if unicodedata.category(c) in allowed_categories else '#'

    clean = ''.join(map(filter_function, clean)).replace('#', ' ')

    # Últimos retoques
    clean = re.sub(r'\s+', ' ', clean).lower().strip()

    # Filtrar palavras muito utilizadas que nao representam muita coisa nesse contexto
    words = (w for w in clean.split() if not w in IGNORE_WORDS and not w.isdigit())

    stemmer = nltk.stem.snowball.PortugueseStemmer()

    return ' '.join(itertools.imap(stemmer.stem, words))

def build_authors_matrix(storage, buckets):
    # gerar uma matriz n * 2 onde as linhas representam os indices dos autores no set de autores,
    # a primeira coluna indica o indice do primeiro documento do autor em questao, e a segunda,
    # o indice de seu ultimo documento
    authors_labels = {}

    for bucket in buckets:

        for label in storage.list_labels(bucket):
            md = storage.get_metadata(bucket, label)
            author = md['orador']
            authors_labels.setdefault(author, []).append(label)

    # Remover todos os deputados com apenas um discurso
    for author in authors_labels.keys():
        if len(authors_labels[author]) < 2:
            del authors_labels[author]

    authors = sorted(authors_labels.keys())

    authors_matrix = []
    document_list = []

    for author in authors:
        if not authors_matrix:
            # matrizes utilizam índice 1-
            first = 1
        else:
            first = authors_matrix[-1][1] + 1

        # documentos deste autor
        docs = authors_labels[author]

        # A lista de documentos deve estar ordenada de acordo com os autores!
        document_list.extend(docs)

        authors_matrix.append((first, first + len(docs) - 1))

    return document_list, authors_matrix


# inicializar o armazenamento
storage = PTOFS()
storage.list_buckets()

# primeiros nbuckets disponiveis
nbuckets = 5
buckets = list(itertools.islice(storage.list_buckets(), 0, nbuckets))

# Gerar uma DTM a partir de todos os documentos nos buckets selecionados
documents, authors = build_authors_matrix(storage, buckets)

print('Processando {0} documentos...'.format(len(documents)))

def load_and_prepare_document(label):
    bucket = label.split(':')[0]
    doc = storage.get_stream(bucket, label)
    return prepare_document(doc)

# carregar documentos e gerar uma dtm
docs = itertools.imap(load_and_prepare_document, documents)

dtm = dtm_as_dataframe(docs, labels=documents)

# remover da DTM palavras pouco utilizadas

class WordFrequencyHelper(object):
    def __init__(self, min=2, max=float('inf')):
        self.min, self.max = min, max
        self.unused = []
        self.frequent = []

    def __call__(self, series):
        s = series.sum()
        if s < self.min:
            self.unused.append(series.name)
        if s > self.max:
            self.frequent.append(series.name)

# identificar e remover palavras usadas menos de (7 * nbuckets) vezes
# note que esse numero e completamente arbitrario e eu nao faco ideia
# do que estou fazendo!
fd = WordFrequencyHelper(min=(7*nbuckets))

dtm.apply(fd, 0)
dtm.drop(fd.unused, axis=1, inplace=True)

print('Aplicando vonmon a {0} documentos, {1} palavras...'.format(
    len(dtm.index), len(dtm.columns),
))

# interfacear com R :)
this_dir = os.path.dirname(__file__)

rpy2.robjects.r('setwd("{0}")'.format(this_dir))
vonmon = rpy2.robjects.r("source('../r/ExpAgendVMVA.R')")[0]

# converter nossa matriz de autores para uma matriz r
authors = pandas.DataFrame(numpy.matrix(authors))

rauthors = pandas.rpy.common.convert_to_r_matrix(authors)
rdtm = pandas.rpy.common.convert_to_r_matrix(dtm)

topics = vonmon(term_doc=rdtm, authors=rauthors, n_cats=70, verbose=True, kappa=400)
