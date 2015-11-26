# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import re
import unicodedata
import itertools

import nltk
import pyth.document
from pyth.plugins.plaintext.writer import PlaintextWriter

# XXX we should use local relative imports, but we are not in a package yet
from rtfreader import CustomRtf15Reader as Rtf15Reader


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


def extract_rtf_text(rtf_file, encoding='utf-8'):
    rtfdoc = Rtf15Reader.read(rtf_file)

    sanitize_rtf_document(rtfdoc)

    return PlaintextWriter.write(rtfdoc).read().decode(encoding)


def is_simple_char(c):
    return c in {'Lu', 'Ll', 'Nd', 'Zs'}


def transliterate(s, encoding='utf-8'):
    # Do our best to replace special characters (mostly accentuated chars)
    # with their corresponding transliterated simplified chars
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode(encoding)
    return ''.join(filter(is_simple_char, s))


def stem_rtf_file(rtf_file):
    text = extract_rtf_text(rtf_file)

    clean = transliterate(text)

    clean = clean.lower()

    # Remove duplicated, starting and ending spaces
    clean = re.sub(r'\s+', ' ', clean).strip()

    # Reduce words to their stemmed version
    stemmer = nltk.stem.snowball.PortugueseStemmer()

    return ' '.join(itertools.imap(stemmer.stem, clean.split()))
