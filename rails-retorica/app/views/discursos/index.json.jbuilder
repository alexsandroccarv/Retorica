json.array!(@discursos) do |discurso|
  json.extract! discurso, 
  json.url discurso_url(discurso, format: :json)
end