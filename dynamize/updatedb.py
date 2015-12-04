# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import argparse
import hashlib
import shutil

import datetime
import requests
import zipfile
import xmltodict
import pymongo
from StringIO import StringIO
from common import transliterate_like_rails


def timestamp():
    utcnow = datetime.datetime.utcnow()
    return {'created_at': utcnow}


def main(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('-H', '--host', type=unicode, default='localhost')
    parser.add_argument('-P', '--port', type=int, default=27017)
    parser.add_argument('-d', '--database', type=unicode, default='retorica')

    args = parser.parse_args(argv)

    mongo = pymongo.MongoClient(args.host, args.port)

    database = getattr(mongo, args.database)

    deputados_collection = database.deputados
    partidos_collection = database.partidos
    uf_collection = database.unidade_federativas

    deputados_file = StringIO(deputados_request.content)

    deputados_zip_file = zipfile.ZipFile(deputados_file)

    deputados_xml = deputados_zip_file.open(u'Deputados.xml')

    deputados = xmltodict.parse(deputados_xml.read())

    deputados = deputados['orgao']['Deputados']['Deputado']

    for deputado in deputados:

        url = (u'http://www.camara.gov.br/SitCamaraWS/'
               u'Deputados.asmx/ObterDetalhesDeputado'
               u'?ideCadastro={ideCadastro}'
               u'&numLegislatura={numLegislatura}').format(**deputado)

        print(u'[{cur}/{max}] {url}'.format(cur=deputados.index(deputado) + 1,
                                            max=len(deputados),
                                            url=url))

        detalhes_deputado_request = requests.get(url)

        detalhes_deputado = xmltodict.parse(detalhes_deputado_request.content)

        detalhes_deputado = detalhes_deputado['Deputados']['Deputado']

        # XXX Is this really necessary? Will the server ever respond with more than one
        # "Deputado" in this list?
        if not isinstance(detalhes_deputado, (tuple, list)):
            detalhes_deputado = [detalhes_deputado]

        for detalhes in detalhes_deputado:

            foto_file_name = '{ideCadastro}.jpg'.format(**detalhes)

            foto_url = ('http://www.camara.gov.br/internet/'
                        'deputado/bandep/{0}').format(foto_file_name)

            foto_path = 'fotos/{0}'.format(foto_file_name)

            foto_request = requests.get(foto_url, stream=True)

            if foto_request.status_code == 200:
                with open(foto_path, 'wb') as foto_file:
                    foto_request.raw.decode_content = True
                    shutil.copyfileobj(foto_request.raw, foto_file)

                with open(foto_path, 'r') as foto_file:
                    foto_fingerprint = hashlib.md5(foto_file.read()).hexdigest()
                    foto_file_size = foto_file.tell()

                foto_content_type = 'image/jpeg'
            else:
                foto_file_size = None
                foto_content_type = None
                foto_fingerprint = None

            uf = uf_collection.update_one(
                {'sigla': detalhes['ufRepresentacaoAtual']}, {
                    '$setOnInsert': timestamp(),
                    '$set': {
                        'sigla': detalhes['ufRepresentacaoAtual'],
                        'updated_at': datetime.datetime.utcnow(),
                    }
                }, upsert=True)

            partido = partidos_collection.update_one(
                {'sigla': detalhes['partidoAtual']['sigla']}, {
                    '$setOnInsert': timestamp(),
                    '$set': {
                        'nome': detalhes['partidoAtual']['nome'],
                        'sigla': detalhes['partidoAtual']['sigla'],
                        'updated_at': datetime.datetime.utcnow(),
                    }
                }, upsert=True)

            site = ('http://www.camara.leg.br/internet/'
                    'deputado/Dep_Detalhe.asp?id={ideCadastro}').format(**detalhes)

            # TODO We're missing the `created_at` value. I don't know how to only set it if not already set.
            deputado = deputados_collection.update_one(
                {'ide_cadastro': detalhes['ideCadastro']}, {
                    '$setOnInsert': timestamp(),
                    '$set': {
                        'updated_at': datetime.datetime.utcnow(),
                        'ide_cadastro': detalhes['ideCadastro'],
                        'partido_ids': [partido.upserted_id],
                        'unidade_federativa_ids': [uf.upserted_id],
                        'site_deputado': site,
                        'nome_parlamentar': detalhes['nomeParlamentarAtual'],
                        'sexo': detalhes['sexo'],
                        'email': detalhes['email'],
                        'situacao': detalhes['situacaoNaLegislaturaAtual'],
                        'legislatura': detalhes['numLegislatura'],
                        'clean_name': transliterate_like_rails(detalhes['nomeParlamentarAtual']),
                        'foto_file_name': foto_file_name,
                        'foto_file_size': foto_file_size,
                        'foto_fingerprint': foto_fingerprint,
                        'foto_content_type': foto_content_type,
                        'foto_updated_at': datetime.datetime.utcnow(),
                    }
                }, upsert=True)


if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv[1:]) or 0)