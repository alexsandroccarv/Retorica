# -*- coding: utf-8 -*-

# Scrapy settings for kingsnake project
#
# For simplicity, this file contains only the most important settings by
# default. All the other settings are documented here:
#
#     http://doc.scrapy.org/en/latest/topics/settings.html
#

from __future__ import unicode_literals
import os.path


BOT_NAME = 'kingsnake'

SPIDER_MODULES = ['kingsnake.spiders']
NEWSPIDER_MODULE = 'kingsnake.spiders'

# Crawl responsibly by identifying yourself (and your website) on the user-agent
USER_AGENT = 'kingsnake (+http://labhackercd.net)'

DEFAULT_REQUEST_HEADERS = {
    'Host': 'www.camara.gov.br',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Encoding': 'gzip, deflate, sdch',
    'Accept-Language': 'pt-BR,pt;q=0.8,en-US;q=0.6,en;q=0.4',
}

# TeorDiscursoPipeline should come first so that the downloaded files are persisted
# in the database by DiscursosDbLitePipeline.
ITEM_PIPELINES = {
    'kingsnake.pipelines.TeorDiscursoPipeline': 1,
    'kingsnake.pipelines.DiscursoDbLitePipeline': 2,
    'kingsnake.pipelines.DeputadosDbLitePipeline': 3,
}

DOWNLOADER_MIDDLEWARES = {
    'scrapy.downloadermiddlewares.defaultheaders.DefaultHeadersMiddleware': 1,
    'kingsnake.middleware.RandomUserAgentMiddleware': 401,
}

RETRY_HTTP_CODES = [500, 502, 503, 504, 400, 403, 408]


def path_from_here(*args):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), *args))


FILES_STORE = path_from_here('..', 'files')

SQLITE_DSN = 'sqlite://{path}'.format(path=path_from_here('..', 'db.sqlite'))
