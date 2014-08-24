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

import pandas
from sklearn.feature_extraction.text import CountVectorizer

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

    # A esta altura, temos apenas uma lista de palavras sem acentos.
    words = clean.split()

    # Agora precisamos remover palavras muito utilizadas, como preposições e
    # artigos.
    #
    # TODO O que mais precisamente deveríamos remover? A descrição original do
    # retórica diz " Também foram removidas palavras muito utilizadas e palavras
    # pouco utilizadas" mas o que exatamente isso quer dizer?

    stemmer = nltk.stem.snowball.PortugueseStemmer()

    return ' '.join(itertools.imap(stemmer.stem, (w for w in clean.split())))

# pegar o primeiro bucket do sistema de armazenamento
storage = PTOFS()
storage.list_buckets()

bucket = next(storage.list_buckets())

# Gerar uma DTM a partir de todos os documentos no bucket selecionado
labels = storage.list_labels(bucket)

# gerar uma matriz n * 2 onde as linhas representam os indices dos autores no set de autores,
# a primeira coluna indica o indice do primeiro documento do autor em questao, e a segunda,
# o indice de seu ultimo documento
authors_labels = {}

for label in labels:
    md = storage.get_metadata(bucket, label)
    docs = authors_labels.setdefault(md['orador'], [])
    docs.append(label)

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

def load_and_prepare_document(label):
    doc = storage.get_stream(bucket, label)
    return prepare_document(doc)

docs = itertools.imap(load_and_prepare_document, document_list)

dtm = dtm_as_dataframe(docs, labels=labels)

import pandas.rpy.common
import rpy2.robjects
from rpy2.robjects.packages import importr
import os.path

this_dir = os.path.dirname(__file__)

rpy2.robjects.r('setwd("{0}")'.format(this_dir))
vonmon = rpy2.robjects.r("source('../r/ExpAgendVMVA.R')")[0]

authors_matrix = pandas.DataFrame(numpy.matrix(authors_matrix))

r_authors = pandas.rpy.common.convert_to_r_matrix(authors_matrix)
r_dtm = pandas.rpy.common.convert_to_r_matrix(dtm)

vonmon(term_doc=r_dtm, authors=r_authors, n_cats=70, verbose=True, kappa=400)
