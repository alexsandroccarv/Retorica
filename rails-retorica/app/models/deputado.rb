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
  field :uf, type: String
  field :email, type: String

  has_many :filiacao_partidaria
  belongs_to :unidade_federativa

end
