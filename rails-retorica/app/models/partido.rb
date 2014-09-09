# -*- encoding : utf-8 -*-
class Partido
  include Mongoid::Document
  include Mongoid::Timestamps


  field :sigla, type: String
  field :nome, type: String
  has_many :filiacao_partidarias
end