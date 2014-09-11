class UnidadeFederativa
  include Mongoid::Document

  field :sigla, type: String
  has_and_belongs_to_many :deputados
end