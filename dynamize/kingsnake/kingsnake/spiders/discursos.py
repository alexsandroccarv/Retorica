# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import datetime
import xmltodict
from scrapy import log
from scrapy.contrib import spiders
from kingsnake.items import Discurso, Sessao


def ensure_list(i):
    if not isinstance(i, (list, tuple)):
        i = [i]
    return i


class DiscursosSpider(spiders.XMLFeedSpider):
    name = 'discursos'
    allowed_domains = ['www.camara.gov.br']
    itertag = 'sessoesDiscursos'
    iterator = 'iternodes'

    def __init__(self, *args, **kwargs):
        self._force_start_urls = kwargs.pop('start_urls', None)
        super(DiscursosSpider, self).__init__()

    def start_requests(self):
        urls = self._force_start_urls or self._start_urls()
        return (self.make_requests_from_url(u) for u in urls)

    def _start_urls(self):
        """Yield urls for `start_requests`.
        """
        url = (
            'http://www.camara.gov.br/sitcamaraws/'
            'SessoesReunioes.asmx/ListarDiscursosPlenario'
            '?dataIni={ini}&dataFim={end}&codigoSessao=&'
            'parteNomeParlamentar=&siglaPartido=&siglaUF='
        )

        # TODO FIXME we should somehow parametrize all of this
        ini = datetime.date(2011, 1, 1)
        end = datetime.date.today()
        step = datetime.timedelta(days=90)
        date_format = r'%d/%m/%Y'

        urls = []

        while ini <= end:
            it_end = min(ini + step, end)

            yield url.format(
                ini=ini.strftime(date_format),
                end=it_end.strftime(date_format),
            )

            ini = it_end + datetime.timedelta(days=1)

    def parse_node(self, response, node):
        data = xmltodict.parse(node.extract())

        sessoes = data.pop('sessoesDiscursos').pop('sessao')

        for sessao in ensure_list(sessoes):
            fases = sessao.pop('fasesSessao').pop('faseSessao')

            sessao['data'] = self._safely_parse_datetime(sessao.get('data'))

            #yield Sessao(**sessao)

            for fase in ensure_list(fases):

                discursos = fase.pop('discursos').pop('discurso')

                for discurso in ensure_list(discursos):
                    discurso['sessao'] = sessao.get('codigo')

                    # copy the dict, just for the sake of it
                    discurso['faseSessao'] = dict(fase)

                    discurso['horaInicioDiscurso'] = \
                        self._safely_parse_datetime(discurso.get('horaInicioDiscurso'))

                    discurso['numeroInsercao'] = self._safely_parse_int(
                        discurso.get('numeroInsercao'))

                    discurso['numeroQuarto'] = self._safely_parse_int(
                        discurso.get('numeroQuarto'))

                    discurso.get('orador')['numero'] = self._safely_parse_int(
                        discurso.get('orador').get('numero'))

                    # FIXME Ignore this field for now
                    del discurso['txtIndexacao']

                    yield Discurso(**discurso)

    def _safely_parse_datetime(self, s):
        if s is None:
            return

        fmt = r'%d/%m/%Y'

        # If it has a space, it has time
        if len(s.split()) > 1:
            fmt += r' %H:%M:%S'

        try:
            return datetime.datetime.strptime(s, fmt)
        except ValueError:
            self.log("Failed to parse date '{0}' using format '{1}'".format(
                s, fmt
            ), log.ERROR)

    def _safely_parse_int(self, s):
        try:
            return int(s)
        except ValueError:
            self.log(
                "Failed to parse '{0}' as an integer".format(s), log.ERROR)
