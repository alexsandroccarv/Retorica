# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import re
import itertools
import unicodedata

import nltk.stem.snowball

from .transliterate import transliterate_like_rails


def remove_punctuation(string):
    """Do our best to remove punctuation and stuff that don't compose words
    """
    allowed_categories = set(('Lu', 'Ll', 'Nd', 'Zs'))

    filter_function = lambda c: c if unicodedata.category(c) in allowed_categories else '#'

    clean = ''.join(map(filter_function, clean)).replace('#', ' ')

    # We don't want no extra spaces
    return re.sub(r'\s+', ' ', clean).lower().strip()


def stemmify_text(text):
    # Do our best to replace special characters (mostly accentuated chars)
    # with their corresponding transliterated simplified chars

    #clean = unicodedata.normalize('NFKD', plaintext).encode('ascii', 'ignore').decode('utf-8')
    clean = transliterate_like_rails(plaintext)

    clean = remove_punctuation(clean)

    # Reduce words to their stemmed version
    stemmer = nltk.stem.snowball.PortugueseStemmer()

    return ' '.join(itertools.imap(stemmer.stem, clean.split()))


def strip_deputy_name(name):
    # strip AKIRA OTSUBO (PRESIDENTE)
    name = re.sub(r'\s*\([^\)]+\)\s*$', '', name)

    # strip ALGUEM, ALGUMA COISA
    name = re.sub(r'\s*,.*$', '', name)

    # strip ALGUEM - PARLAMENTAR JOVEM
    # but don't touch AKIRA-TO
    name = re.sub(r'\s+-.*$', '', name)

    return name
