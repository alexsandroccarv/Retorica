# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from kingsnake.pipelines import DiscursosMongoDBPipeline


def speech_collection_from_command(command, crawler):
    """Returns the *Mongo collection* where the speeches are stored.
    """
    class Struct(object): pass

    spider = Struct()
    spider.settings = command.crawler_process.settings
    spider.crawler = crawler

    pipeline = DiscursosMongoDBPipeline()
    pipeline.open_spider(spider)

    return pipeline.collection
