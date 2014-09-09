class UnidadeFederativa
  include Mongoid::Document

  field :sigla, type: String
  embeds_many :deputados
end