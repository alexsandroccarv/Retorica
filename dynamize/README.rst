ws.py
=====

O script **ws.py** varre o serviço de dados abertos da câmara coletando os
conteúdos dos discursos nos períodos especificados (atualmente desde o dia
2/2/2011) e os salva numa estrutura de diretórios gerada com o pacote
Pairtree, com os metadados relevantes associados.

Em breve escreverei um script com exemplos de como ler essa estrutura e
extrair os metadados.

::

    $ pip install suds ofs pairtree
    $ python ws.py
