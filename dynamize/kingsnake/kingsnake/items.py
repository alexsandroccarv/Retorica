# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import scrapy


class Sessao(scrapy.Item):
    codigo = scrapy.Field()
    data = scrapy.Field()
    numero = scrapy.Field()
    tipo = scrapy.Field()


class Discurso(scrapy.Item):
    # unique key
    _id = scrapy.Field()

    # this will be a dict
    orador = scrapy.Field()
    horaInicioDiscurso = scrapy.Field()
    numeroQuarto = scrapy.Field()
    numeroInsercao = scrapy.Field()
    sumario = scrapy.Field()

    # this will be the session code
    sessao = scrapy.Field()

    # this will also be a dict
    faseSessao = scrapy.Field()

    # The speech content is saved as a file
    files = scrapy.Field()
