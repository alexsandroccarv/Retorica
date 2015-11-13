# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import sys
import base64
import datetime
import logging
import os.path
import argparse

import suds
import pymongo
from clint.textui import indent
from common import puts


def ensure_iterator(obj):
    if isinstance(obj, list):
        iterable = obj
    else:
        iterable = [obj]
    return iter(iterable)


def readabledate(dt):
    return dt.strftime('%Y/%m/%d')


def main(argv):

    parser = argparse.ArgumentParser()

    parser.add_argument('-H', '--host', type=unicode, default='localhost')
    parser.add_argument('-P', '--port', type=int, default=27017)
    parser.add_argument('-d', '--database', type=unicode, default='retorica')
    parser.add_argument('-c', '--collection', type=unicode, default='speeches')

    args = parser.parse_args(argv[1:])

    mongo = pymongo.MongoClient(args.host, args.port)
    database = getattr(mongo, args.database)
    collection = getattr(database, args.collection)

    service_url = 'http://www.camara.gov.br/SitCamaraWS/SessoesReunioes.asmx?wsdl'

    client = suds.client.Client(service_url, cachingpolicy=1)

    # Date format used by the service
    date_format = '%d/%m/%Y'

    # A data inicial dos discursos
    start_date = datetime.date(2011, 2, 2)

    # Chunk size in days
    # Note that the service only accepts chunks of 360 days max
    chunk_size = datetime.timedelta(days=360)

    sd = start_date
    today = datetime.date.today()

    count = 0

    while sd <= today:
        ed = min(sd + chunk_size, today)

        puts('Obtaining speeches given between {sd} and {ed} é'.format(
            sd=readabledate(sd), ed=readabledate(ed)))

        sessoes = client.service.ListarDiscursosPlenario(
            dataIni=sd.strftime(date_format), dataFim=ed.strftime(date_format))

        first_session = next(ensure_iterator(sessoes.sessoesDiscursos.sessao))
        first = next(ensure_iterator(first_session.fasesSessao.faseSessao))
        first = next(ensure_iterator(first.discursos.discurso))

        puts('   {0} sessões obtidas'.format(len(sessoes.sessoesDiscursos.sessao)))
        puts('   Primeira sessão: {0}'.format(first_session.numero))
        puts('   Primeiro discurso: {0}'.format(first.horaInicioDiscurso))

        previous_count = count

        for sessao in ensure_iterator(sessoes.sessoesDiscursos.sessao):

            try:
                dt = datetime.date.strptime(sessao.data.strip(), '%Y/%m/%d')
            except:
                dt = None

            try:
                numero = int(sessao.numero)
            except:
                numero = sessao.numero

            s = {
                'codigo': unicode(sessao.codigo).strip(),
                'data': dt,
                'numero': numero,
                'tipo': unicode(sessao.tipo.strip()),
            }

            database.sessoes.insert(s)

            for fase in ensure_iterator(sessao.fasesSessao.faseSessao):

                f = {
                    'codigo': unicode(fase.codigo.strip()),
                    'descricao': unicode(fase.descricao.strip()),
                }

                for discurso in ensure_iterator(fase.discursos.discurso):

                    count += 1

                    wsid = ':'.join([
                        sessao.codigo.strip(),
                        discurso.orador.numero,
                        discurso.numeroQuarto,
                        discurso.numeroInsercao,
                    ])

                    if not database.discursos.find_one({'wsid': wsid}):
                        puts('Obtendo discurso {0}'.format(wsid))

                        try:
                            teor = client.service.obterInteiroTeorDiscursosPlenario(
                                codSessao=sessao.codigo.strip(),
                                numOrador=discurso.orador.numero,
                                numQuarto=discurso.numeroQuarto,
                                numInsercao=discurso.numeroInsercao,
                            )
                        except:
                            puts("Failed to get {0}".format(wsid))
                            continue

                        teor = teor.sessao

                        conteudo = base64.b64decode(teor.discursoRTFBase64)

                        dt = teor.horaInicioDiscurso
                        if dt:
                            dt = datetime.datetime.strptime(dt, '%d/%m/%Y %H:%M:%S')

                        d = {
                            'wsid': wsid,
                            'sessao': s['codigo'],
                            'fase_sessao': f,
                            'autor': unicode(teor.nome).strip(),
                            'partido': unicode(teor.partido).strip(),
                            'proferido_em': dt,
                            'numero_orador': int(discurso.orador.numero),
                            'numero_quarto': int(discurso.numeroQuarto),
                            'numero_insercao': int(discurso.numeroInsercao),
                            'sumario': unicode(discurso.sumario.strip()),
                            'conteudo': conteudo,
                        }
                        database.discursos.insert(d)
        else:
            puts('   Última sessão: {0}'.format(sessao.numero))
            puts('   Último discurso: {0}'.format(discurso.horaInicioDiscurso))

        puts('{0} discursos obtidos no período de {1} a {2}'.format(
            count - previous_count, sd.strftime(date_format), ed.strftime(date_format)
        ))

        sd = ed + datetime.timedelta(days=1)

    puts('{0} discursos obtidos'.format(count))


if __name__ == '__main__':
    sys.exit(main(sys.argv) or 0)
