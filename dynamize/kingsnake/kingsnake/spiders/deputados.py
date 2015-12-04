# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from scrapy import spiders


class DeputadosSpider(spiders.XMLFeedSpider):
    name = 'deputados'
    allowed_domains = ['www.camara.gov.br']
    itertag = 'sessoesDiscursos'
    iterator = 'iternodes'

    start_urls = [('http://www2.camara.leg.br/'
                   'transparencia/dados-abertos/'
                   'dados-abertos-legislativo/'
                   'webservices/deputados/deputados')]

    def parse(self, response):
        import pdb; pdb.set_trace()
