json.id_cadastro @deputado._id
json.extract! @deputado, :id_parlamentar, :nome_parlamentar, :sexo, :email
json.partido @deputado.partidos.first.sigla
json.uf @deputado.unidade_federativa.sigla

