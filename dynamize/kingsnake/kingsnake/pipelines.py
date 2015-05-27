# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import base64
from StringIO import StringIO as BytesIO
from scrapy_mongodb import MongoDBPipeline

import xmltodict
from scrapy import log
from scrapy.http import Request
from scrapy.contrib.pipeline.files import FilesPipeline, FileException
from scrapy.utils.misc import md5sum


_null = object()


def item_is_speech(item):
    """Dumb method to guess if the given *item* is a Speech
    """
    return item.get('sessao', _null) is not _null


def item_is_session(item):
    """Dumb method to guess if the given *item* is a Session
    """
    return item.get('codigo', _null) is not _null


class ItemSpecificPipeline(object):
    """Mixin for pipelines which are specific to some items. You just
    reimplement `should_process_item` to return weather or not the specific
    *item* should be processed.
    """

    def should_process_item(self, item, spider):
        return True

    def process_item(self, item, spider):
        if not self.should_process_item(item, spider):
            return item
        else:
            return super(ItemSpecificPipeline, self).process_item(item, spider)


class NamedCollectionMongoDBPipeline(MongoDBPipeline):
    """A `MongoDBPipeline` whose `collection` is independent from the
    configured `MONGODB_COLLECTION`. Instead, it's specified through
    `collection_name`.
    """

    # Specify the collection name!
    collection_name = None

    def configure(self, *args, **kwargs):
        super(NamedCollectionMongoDBPipeline, self).configure(*args, **kwargs)

        if self.collection_name is not None:
            self.config['collection'] = self.collection_name


class DiscursosMongoDBPipeline(ItemSpecificPipeline,
                               NamedCollectionMongoDBPipeline):

    null = object()
    collection_name = 'discursos'

    def should_process_item(self, item, spider):
        return item_is_speech(item)


class SessoesMongoDBPipeline(ItemSpecificPipeline,
                             NamedCollectionMongoDBPipeline):

    null = object()
    collection_name = 'sessoes'

    def should_process_item(self, item, spider):
        return item_is_session(item)


class RetryException(Exception):
    def __init__(self, original_exc, *args, **kwargs):
        self.original_exc = original_exc
        super(RetryException, self).__init__(*args, **kwargs)


class RetryMediaPipelineMixin(object):

    max_retry_times = 3

    def media_downloaded(self, response, request, info):
        try:
            return super(RetryMediaPipelineMixin, self).media_downloaded(
                response, request, info)
        except FileException as exc:
            if self._should_retry(exc, response, request, info):
                raise RetryException(exc)
            else:
                raise

    def _should_retry(self, exception, response, request, info):
        return True

    def _process_request(self, request, info):
        pass

    def _retry(self, request, reason, info):
        retries = request.meta.get('retry_times', 0) + 1

        if retries <= self.max_retry_times:
            log.msg(format="Retrying %(request)s (failed %(retries)d times): %(reason)s",
                    level=log.DEBUG, spider=info.spider, request=request, retries=retries, reason=reason)
            retryreq = request.copy()
            retryreq.meta['retry_times'] = retries
            retryreq.dont_filter = True
            #retryreq.priority = request.priority + self.priority_adjust
            #return retryreq

            return self._process_request()
        else:
            log.msg(format="Gave up retrying %(request)s (failed %(retries)d times): %(reason)s",
                    level=log.DEBUG, spider=info.spider, request=request, retries=retries, reason=reason)


class TeorDiscursoPipeline(ItemSpecificPipeline, FilesPipeline):

    def should_process_item(self, item, spider):
        return item_is_session(item)

    def get_media_requests(self, item, info):

        url = ('http://www.camara.gov.br/SitCamaraWS/'
                'SessoesReunioes.asmx/obterInteiroTeorDiscursosPlenario'
                '?codSessao={sessao}&numOrador={orador}'
                '&numQuarto={quarto}&numInsercao={insercao}')

        url = url.format(sessao=item.get('sessao'),
                         orador=item.get('orador').get('numero'),
                         quarto=item.get('numeroQuarto'),
                         insercao=item.get('numeroInsercao'))

        return [Request(url)]

    def file_downloaded(self, response, request, info):

        # The downloaded file is a XML file which stores the actual RTF file as
        # a base64 encoded string. Here we extract and decode that value.

        path = self.file_path(request, response=response, info=info)

        rdata = xmltodict.parse(response.body)

        content = rdata.get('sessao').get('discursoRTFBase64')

        content = base64.b64decode(content)

        buf = BytesIO(content)

        self.store.persist_file(path, buf, info)

        checksum = md5sum(buf)

        return checksum

    def item_completed(self, results, item, info):
        if isinstance(item, dict) or self.FILES_RESULT_FIELD in item.fields:
            item[self.FILES_RESULT_FIELD] = [x for ok, x in results if ok]
        return item

    def file_path(self, request, response=None, info=None):
        # XXX Original `file_path` implementation appends the request's query
        # string into the file name, and we don't like that, so we're
        # overriding it and hacking around it.
        path = super(TeorDiscursoPipeline, self).file_path(request,
                                                           response, info)
        fname, ext = path.split('.', 1)
        return '.'.join([fname, 'rtf'])
