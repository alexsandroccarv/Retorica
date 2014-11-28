# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from kingsnake.pipelines import DiscursosMongoDBPipeline


def speech_collection_from_command(command):
    """Returns the *Mongo collection* where the speeches are stored.
    """
    class Struct(object): pass

    crawler = Struct()
    crawler.settings = command.crawler_process.settings

    pipeline = DiscursosMongoDBPipeline(crawler)

    return pipeline.collection
