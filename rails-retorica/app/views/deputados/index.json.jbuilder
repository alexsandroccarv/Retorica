json.array!(@deputados) do |deputado|
  json.id_cadastro deputado._id
  json.extract! deputado, :url_parlamentar, :nome_parlamentar, :sexo, :email
  json.partido deputado.partidos.first.sigla
  json.uf deputado.unidade_federativa.sigla
  json.foto deputado.foto.url
  json.dados_deputado deputado_url(deputado, format: :json)
end