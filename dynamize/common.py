# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import unicodedata
import unihandecode
from clint.textui import puts as unsafe_puts


def puts(s, *args, **kwargs):
    return unsafe_puts(s.encode('utf-8'), *args, **kwargs)


def transliterate_like_rails(string):
    """Our best attempt to do the same thing as ActiveSupport::Inflector::transliterate
    """
    string = unicodedata.normalize('NFKC', string)
    string = unihandecode.unidecode(string)
    return string.encode('ascii', 'replace').decode('utf-8')
