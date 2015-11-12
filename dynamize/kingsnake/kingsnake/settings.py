# -*- coding: utf-8 -*-

# Scrapy settings for kingsnake project
#
# For simplicity, this file contains only the most important settings by
# default. All the other settings are documented here:
#
#     http://doc.scrapy.org/en/latest/topics/settings.html
#

BOT_NAME = 'kingsnake'

SPIDER_MODULES = ['kingsnake.spiders']
NEWSPIDER_MODULE = 'kingsnake.spiders'

# Crawl responsibly by identifying yourself (and your website) on the user-agent
#USER_AGENT = 'kingsnake (+http://www.yourdomain.com)'

ITEM_PIPELINES = [
    'kingsnake.pipelines.TeorDiscursoPipeline',
    'kingsnake.pipelines.SessoesMongoDBPipeline',
    'kingsnake.pipelines.DiscursosMongoDBPipeline',
]

MONGODB_URI = 'mongodb://localhost:27017'
MONGODB_DATABASE = 'retorica'
MONGODB_UNIQUE_KEY = '_id'

DOWNLOADER_MIDDLEWARES = {
    'kingsnake.middleware.RandomUserAgentMiddleware': 401,
}

RETRY_HTTP_CODES = [500, 502, 503, 504, 400, 403, 408]


import os.path

def here(*args):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), *args))

FILES_STORE = here('files')
