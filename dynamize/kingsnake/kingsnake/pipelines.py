# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import base64

import dblite
import xmltodict
from scrapy.http import Request
from scrapy.pipelines.files import FilesPipeline

from kingsnake.items import Discurso


class DbLitePipeline(object):

    item_class = None
    table_name = None

    def open_spider(self, spider):
        self.crawler = spider.crawler
        self.settings = spider.settings

        dsn = self.settings.get('SQLITE_DSN')

        table_name = self.table_name
        if table_name is None:
            table_name = self.item_class.__class__.__name__.lower()

        self.dsn = ':'.join([dsn, table_name])
        self.dblite = dblite.open(self.item_class, self.dsn, autocommit=True)

    def close_spider(self, spider):
        self.dblite.commit()
        self.dblite.close()

    def process_item(self, item, spider):
        if isinstance(item, self.item_class):
            item = self._do_process_item(item)
        return item

    def _do_process_item(self, item):

        # If it has an *_id* we should just update it and move on.
        if item.get('_id') is not None:
            self.dblite.update(item)
            return item

        old = self.dblite.get_one({
            'faseSessao': item['faseSessao'],
            'numeroOrador': item['numeroOrador'],
            'numeroQuarto': item['numeroQuarto'],
            'numeroInsercao': item['numeroInsercao'],
        })

        if not old:
            self.dblite.put(item)
        else:
            old.update(item)
            self.dblite.update(old)

        # FIXME What should we return? ...
        # ... The original item or the newly inserted item? Pls help.
        return item


class DiscursoDbLitePipeline(DbLitePipeline):
    item_class = Discurso
    table_name = 'discursos'


class TeorDiscursoPipeline(FilesPipeline):

    def process_item(self, item, spider):
        # TODO We should give users a way to "refresh" file contents.
        if isinstance(item, Discurso) and not item.get('files'):
            return super(TeorDiscursoPipeline, self).process_item(item, spider)
        return item

    def get_media_requests(self, item, info):
        url = ('http://www.camara.gov.br/SitCamaraWS/'
                'SessoesReunioes.asmx/obterInteiroTeorDiscursosPlenario'
                '?codSessao={sessao}&numOrador={numeroOrador}'
                '&numQuarto={numeroQuarto}&numInsercao={numeroInsercao}')

        return [Request(url.format(**item))]

    def file_downloaded(self, response, request, info):
        rdata = xmltodict.parse(response.body)
        content = rdata.get('sessao').get('discursoRTFBase64')
        content = base64.b64decode(content)

        request.body = content

        return super(TeorDiscursoPipeline, self).file_downloaded(response, request, info)

    def file_path(self, request, response=None, info=None):
        # XXX Original `file_path` implementation appends the request's query
        # string into the file name, and we don't like that, so we're
        # overriding it and hacking around it.
        path = super(TeorDiscursoPipeline, self).file_path(request, response, info)
        fname, ext = path.split('.', 1)
        return '.'.join([fname, 'rtf'])
