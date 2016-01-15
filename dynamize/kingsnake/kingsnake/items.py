# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import dblite.serializers
import scrapy


class Discurso(scrapy.Item):
    _id = scrapy.Field()

    horaInicioDiscurso = scrapy.Field()

    sessao = scrapy.Field()
    faseSessao = scrapy.Field()

    nomeOrador = scrapy.Field()
    numeroOrador = scrapy.Field()
    partidoOrador = scrapy.Field()
    ufOrador = scrapy.Field()

    numeroQuarto = scrapy.Field()
    numeroInsercao = scrapy.Field()

    sumario = scrapy.Field()

    files = scrapy.Field(dblite_serializer=dblite.serializers.CompressedJsonSerializer)


class Deputado(scrapy.Item):
    _id = scrapy.Field()

    uf = scrapy.Field()
    nome = scrapy.Field()
    partido = scrapy.Field()
    ide_cadastro = scrapy.Field()
    num_legislatura = scrapy.Field()
