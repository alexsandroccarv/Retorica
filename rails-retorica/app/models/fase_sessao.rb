class FaseSessao
  include Mongoid::Document

  field :codigo, type: String
  field :descricao, type: String
  embedded_in :discurso
end
