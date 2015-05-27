# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import os
from scrapy.command import ScrapyCommand
from scrapy.exceptions import UsageError
from kingsnake.utils import speech_collection_from_command


class Command(ScrapyCommand):

    requires_project = True

    def syntax(self):
        return "[options]"

    def short_desc(self):
        return "Download missing speech files"

    def add_options(self, parser):
        ScrapyCommand.add_options(self, parser)

        parser.add_option("-o", "--output", metavar="FILE",
                          help="dump scraped items into FILE (use - for stdout)")
        parser.add_option("-t", "--output-format", metavar="FORMAT",
                          help="format to use for dumping items with -o")

    def process_options(self, args, opts):
        ScrapyCommand.process_options(self, args, opts)

        if opts.output:
            if opts.output == '-':
                self.settings.set('FEED_URI', 'stdout:', priority='cmdline')
            else:
                self.settings.set('FEED_URI', opts.output, priority='cmdline')
            valid_output_formats = (
                list(self.settings.getdict('FEED_EXPORTERS').keys()) +
                list(self.settings.getdict('FEED_EXPORTERS_BASE').keys())
            )
            if not opts.output_format:
                opts.output_format = os.path.splitext(opts.output)[1].replace(".", "")
            if opts.output_format not in valid_output_formats:
                raise UsageError("Unrecognized output format '%s', set one"
                                 " using the '-t' switch or as a file extension"
                                 " from the supported list %s" % (opts.output_format,
                                                                  tuple(valid_output_formats)))
            self.settings.set('FEED_FORMAT', opts.output_format, priority='cmdline')

    def run(self, args, opts):
        spname = 'discursos'

        crawler = self.crawler_process.create_crawler()

        start_urls = self._start_urls(crawler)

        spargs = dict(start_urls=start_urls)

        spider = crawler.spiders.create(spname, **spargs)
        crawler.crawl(spider)
        self.crawler_process.start()

    def _start_urls(self, crawler):
        # XXX To get the start urls, we'll query every document with missing
        # files in the database. We do that through our pipeline's connection
        # *just because we can*.
        collection = speech_collection_from_command(self, crawler)

        speeches = collection.find({
            '$or': [
                {'files': {'$exists': False}},
                {'files': []},
            ],
        })

        return [self._speech_url(s) for s in speeches]

    def _speech_url(self, item):
        url = ('http://www.camara.gov.br/SitCamaraWS/'
                'SessoesReunioes.asmx/obterInteiroTeorDiscursosPlenario'
                '?codSessao={sessao}&numOrador={orador}'
                '&numQuarto={quarto}&numInsercao={insercao}')
        return url.format(sessao=item.get('sessao'),
                          orador=item.get('orador').get('numero'),
                          quarto=item.get('numeroQuarto'),
                          insercao=item.get('numeroInsercao'))
