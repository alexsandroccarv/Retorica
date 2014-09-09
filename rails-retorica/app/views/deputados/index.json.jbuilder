json.array!(@deputados) do |deputado|
  json.extract! deputado, :nome, :partido, :sexo, :uf, :email
  json.url deputado_url(deputado, format: :json)
end