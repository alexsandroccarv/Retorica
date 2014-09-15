json.array!(@deputados) do |deputado|
  json.extract! deputado, :nome_parlamentar, :sexo, :email
  json.partido deputado.partidos.first.sigla
  json.uf deputado.unidade_federativa.sigla
  json.url_interna deputado_url(deputado, format: :json)
end