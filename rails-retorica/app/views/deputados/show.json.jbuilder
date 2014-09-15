json.extract! @deputado, :nome_parlamentar, :sexo, :email
json.data_nascimento @deputado.data_nascimento.strftime('%m/%d/%Y')
json.partido @deputado.partidos.first.sigla
json.uf @deputado.unidade_federativa.sigla

