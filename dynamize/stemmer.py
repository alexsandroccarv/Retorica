# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import re
import unicodedata
import itertools
import os.path

import nltk
import numpy
import pandas
import pandas.rpy.common
import rpy2.robjects

from ofs.local import PTOFS
from pyth.plugins.plaintext.writer import PlaintextWriter
from pyth.plugins.rtf15.reader import Rtf15Reader
from sklearn.feature_extraction.text import CountVectorizer


# Quantidade de buckets que serão lidos. Deixe `0` para utilizar TODOS os
# buckets.
NBUCKETS = 20


IGNORE_WORDS = set([
    # preposições
    'por', 'para', 'a', 'ante', 'ate', 'apos', 'de', 'desde', 'em', 'entre',
    'com', 'sem', 'sob', 'sobre',

    # pronomes
    'eu', 'tu', 'ele', 'ela', 'nos', 'vos', 'eles', 'elas', 'voce', 'me',
    'mim', 'se',

    # artigos e combinações
    'a', 'o', 'ao', 'da', 'do', 'pelo', 'pela', 'as', 'os', 'aos', 'das',
    'dos', 'pelos', 'pelas', 'um', 'uma', 'uns', 'umas'

    # etc
    'que', 'sr', 'sra',
])

IGNORE_STEMS = set([
    'porq', 'nao', 'sao', 'quer', 'ser', 'acontec', 'acord', 'agor',
    'agradec', 'aind', 'algum', 'amanh', 'ano', 'anos', 'apen', 'apoi',
    'apresent', 'aprov', 'assim', 'atend', 'ativ', 'autor', 'vem', 'vam',
    'tud', 'quem', 'quer', 'que', 'onde', 'orador'
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

    return document_list, authors_matrix, authors


# inicializar o armazenamento
storage = PTOFS()
storage.list_buckets()

# primeiros `NBUCKETS` disponiveis
buckets = storage.list_buckets()

if NBUCKETS:
    buckets = itertools.islice(buckets, 0, NBUCKETS)

# Gerar uma DTM a partir de todos os documentos nos buckets selecionados
documents, authors, author_names = build_authors_matrix(storage, buckets)

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

print('Aplicando vonmon a {0} documentos, {1} palavras e {2} autores...'.format(
    len(dtm.index), len(dtm.columns), len(authors)
))

# interfacear com R :)
this_dir = os.path.abspath(os.path.dirname(__file__))

rpy2.robjects.r('setwd("{0}")'.format(this_dir))

# converter nossa matriz de autores para uma matriz r
authors = pandas.DataFrame(numpy.matrix(authors))

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

topics.to_csv(os.path.join(this_dir, 'topics.csv'))

print('topic_words.csv...')

# relação entre palavras e temas
topic_words = pandas.rpy.common.convert_robject(topics[2])
topic_words.to_csv(os.path.join(this_dir, 'topic_words.csv'))

print('Feito!')
