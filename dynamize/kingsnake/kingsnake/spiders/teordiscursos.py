# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from scrapy.spider import Spider
from kingsnake.pipelines import DiscursosMongoDBPipeline


def ensure_list(i):
    if not isinstance(i, (list, tuple)):
        i = [i]
    return i


class TeorDiscursosSpider(Spider):
    name = 'teordiscursos'
    start_urls = ['http://labhackercd.net']

    def _speech_url(self, item):
        url = ('http://www.camara.gov.br/SitCamaraWS/'
                'SessoesReunioes.asmx/obterInteiroTeorDiscursosPlenario'
                '?codSessao={sessao}&numOrador={orador}'
                '&numQuarto={quarto}&numInsercao={insercao}')
        return url.format(sessao=item.get('sessao'),
                          orador=item.get('orador').get('numero'),
                          quarto=item.get('numeroQuarto'),
                          insercao=item.get('numeroInsercao'))

    def parse(self, response):
        pipeline = DiscursosMongoDBPipeline()
        pipeline.open_spider(self)

        speeches = pipeline.collection.find({
            '$or': [
                {'files': {'$exists': False}},
                {'files': {'$size': 0}},
            ],
        })

        for speech in speeches.limit(1000):
            yield speech
