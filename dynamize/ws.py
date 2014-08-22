# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import datetime
import logging
import os.path
import sys

import suds


def get_basic_logger(logger=None, level=logging.INFO):
    if logger is None:
        logger = __name__
    logger = logging.getLogger(logger)
    logger.setLevel(level)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    return logger


def ensure_iterator(obj):
    if isinstance(obj, list):
        iterable = obj
    else:
        iterable = [obj]
    return iter(iterable)


logger = get_basic_logger()

service_url = 'http://www.camara.gov.br/SitCamaraWS\SessoesReunioes.asmx?wsdl'

client = suds.client.Client(service_url, cachingpolicy=1)

cache_file = os.path.abspath(os.path.join(os.path.dirname(__file__), 'cache'))
client.set_options(cache=suds.cache.FileCache(location=cache_file))

date_format = '%d/%m/%Y'

# A data inicial dos discursos
start_date = datetime.date(2011, 2, 2)

# Tamanho do bloco (em dias)
chunk_size = datetime.timedelta(days=360)


# O serviço existe que a diferença entre as datas iniciais e finais seja de no
# máximo 360 dias. Pegaremos, por tanto, os dados em pedaços.

sd = start_date
today = datetime.date.today()

count = 0

while sd <= today:
    ed = min(sd + chunk_size, today)

    logger.info('Obtendo discursos do período entre {sd} e {ed}'.format(
        sd=sd.strftime(date_format), ed=ed.strftime(date_format)))

    sessoes = client.service.ListarDiscursosPlenario(
        dataIni=sd.strftime(date_format), dataFim=ed.strftime(date_format))

    def get_first_discurso(s):
        fs = s.sessoesDiscursos.sessao[0]

    first_session = next(ensure_iterator(sessoes.sessoesDiscursos.sessao))
    first = next(ensure_iterator(first_session.fasesSessao.faseSessao))
    first = next(ensure_iterator(first.discursos.discurso))

    logger.info('   {0} sessões obtidas'.format(len(sessoes.sessoesDiscursos.sessao)))
    logger.info('   Primeira sessão: {0}'.format(first_session.numero))
    logger.info('   Primeiro discurso: {0}'.format(first.horaInicioDiscurso))

    previous_count = count

    for sessao in ensure_iterator(sessoes.sessoesDiscursos.sessao):
        for fase in ensure_iterator(sessao.fasesSessao.faseSessao):
            for discurso in ensure_iterator(fase.discursos.discurso):
                count += 1
    else:
        logger.info('   Última sessão: {0}'.format(sessao.numero))
        logger.info('   Último discurso: {0}'.format(discurso.horaInicioDiscurso))

    logger.info('{0} discursos obtidos no período de {1} a {2}'.format(
        count - previous_count, sd.strftime(date_format), ed.strftime(date_format)
    ))

    sd = ed + datetime.timedelta(days=1)

logger.info('{0} discursos obtidos'.format(count))
