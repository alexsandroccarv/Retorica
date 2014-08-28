Scripts
=======

Scripts experimentais. Em breve os organizaremos melhor. Por enquanto, use
Python 2 e instale tudo isso:

::

    $ pip install ofs pairtree pyth rpy2 scipy scikit-learn nltk numpy pandas

ws.py
-----

O script **ws.py** varre o serviço de dados abertos da câmara coletando os
conteúdos dos discursos nos períodos especificados (atualmente desde o dia
2/2/2011) e os salva numa estrutura de diretórios gerada com o pacote
Pairtree, com os metadados relevantes associados.

Em breve escreverei um script com exemplos de como ler essa estrutura e
extrair os metadados.

::

    $ python ws.py


stemmer.py
----------

O script **stemmer.py** processa os documentos baixados pelo **ws.py**, gera
uma DTM, interfaceia com o R, executa o algoritmo vonmon, salva a humanidade e
domina o mundo.

::

    $ python stemmer.py


Também, é preciso alterar o arquivo `pandas/rpy/common.py` e colocar isso numa
linha específica. Por favor, alguém resolva isso.

::
            try:
                value = VECTOR_TYPES[value_type](value)
            except KeyError:
                # do some magic just because
                x = globals().get('NAMED_VECTOR_TYPES')
                if x is None:
                    x = globals().setdefault('NAMED_VECTOR_TYPES', dict((
                        (t.__name__, v) for (t, v) in VECTOR_TYPES.iteritems()
                    )))
                value = x[value_type.__name__](value)
