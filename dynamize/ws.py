# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import datetime
import logging
import os.path
import base64

import suds
from ofs.local import PTOFS
from ofs.base import BucketExists


def get_basic_logger(logger=None, level=logging.INFO):
    if logger is None:
        logger = __name__
    logger = logging.getLogger(logger)
    logger.setLevel(level)
    return logger


def ensure_iterator(obj):
    if isinstance(obj, list):
        iterable = obj
    else:
        iterable = [obj]
    return iter(iterable)


def ptofs_get_or_claim_bucket(storage, bucket):
    try:
        return storage.claim_bucket(bucket)
    except BucketExists:
        return bucket


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


# Armazenamento em sistema de arquivos
storage = PTOFS()

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

    first_session = next(ensure_iterator(sessoes.sessoesDiscursos.sessao))
    first = next(ensure_iterator(first_session.fasesSessao.faseSessao))
    first = next(ensure_iterator(first.discursos.discurso))

    logger.info('   {0} sessões obtidas'.format(len(sessoes.sessoesDiscursos.sessao)))
    logger.info('   Primeira sessão: {0}'.format(first_session.numero))
    logger.info('   Primeiro discurso: {0}'.format(first.horaInicioDiscurso))

    previous_count = count

    for sessao in ensure_iterator(sessoes.sessoesDiscursos.sessao):

        bucket = ptofs_get_or_claim_bucket(storage, sessao.codigo.strip())

        for fase in ensure_iterator(sessao.fasesSessao.faseSessao):
            for discurso in ensure_iterator(fase.discursos.discurso):
                count += 1

                # Identificador do discurso
                uid = ':'.join([
                    sessao.codigo.strip(),
                    discurso.orador.numero,
                    discurso.numeroQuarto,
                    discurso.numeroInsercao,
                ])

                if not storage.exists(bucket, uid):
                    logger.info('Obtendo discurso {0}'.format(uid))

                    teor = client.service.obterInteiroTeorDiscursosPlenario(
                        codSessao=sessao.codigo.strip(),
                        numOrador=discurso.orador.numero,
                        numQuarto=discurso.numeroQuarto,
                        numInsercao=discurso.numeroInsercao,
                    )
                    teor = teor.sessao

                    conteudo = base64.b64decode(teor.discursoRTFBase64)

                    dt = teor.horaInicioDiscurso
                    if dt:
                        dt = datetime.datetime.strptime(dt, '%d/%m/%Y %H:%M:%S').isoformat()

                    storage.put_stream(bucket, uid, conteudo, {
                        'orador': unicode(teor.nome).strip(),
                        'partido': unicode(teor.partido).strip(),
                        'uf': unicode(teor.uf).strip(),
                        'proferido_em': dt,
                        'codigo_sessao': unicode(sessao.codigo).strip(),
                        'numero_orador': int(discurso.orador.numero),
                        'numero_quarto': int(discurso.numeroQuarto),
                        'numero_insercao': int(discurso.numeroInsercao),
                        'fase_sessao': unicode(fase.descricao).strip(),
                    })
    else:
        logger.info('   Última sessão: {0}'.format(sessao.numero))
        logger.info('   Último discurso: {0}'.format(discurso.horaInicioDiscurso))

    logger.info('{0} discursos obtidos no período de {1} a {2}'.format(
        count - previous_count, sd.strftime(date_format), ed.strftime(date_format)
    ))

    sd = ed + datetime.timedelta(days=1)

logger.info('{0} discursos obtidos'.format(count))
