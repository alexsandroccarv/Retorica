# -*- encoding : utf-8 -*-
class Deputado
  include Mongoid::Document
  include Mongoid::Attributes::Dynamic
  include Mongoid::Timestamps


  field :_id, type: String
  field :situacao_legislatura, type: String
  field :nome_civil, type: String
  field :nome_parlamentar, type: String

  field :data_nascimento, type: Date
  field :sexo, type: String
  field :email, type: String

  has_and_belongs_to_many :partidos
  belongs_to :unidade_federativa

end
