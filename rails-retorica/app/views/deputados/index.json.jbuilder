json.array!(@deputados) do |deputado|

  json.id_cadastro deputado._id
  json.autor deputado.nome_parlamentar
  json.rotulo deputado.enfases.first.tema
  json.enfase deputado.enfases.first.valor
  json.uf deputado.unidade_federativa.first.sigla
  json.partido deputado.partidos.first.sigla
  json.url deputado.site_deputado
  json.foto deputado.foto.url
  json.email deputado.email

end