class Enfase
  include Mongoid::Document
  include Mongoid::Timestamps


  field :tema, type: String
  field :valor, type: Float

  has_and_belongs_to_many :deputado
end
