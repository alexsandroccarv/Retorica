class UnidadeFederativa
  include Mongoid::Document
  include Mongoid::Timestamps

  field :sigla, type: String
  has_and_belongs_to_many :deputados
end