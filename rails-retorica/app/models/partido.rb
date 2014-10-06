# -*- encoding : utf-8 -*-
class Partido
  include Mongoid::Document
  include Mongoid::Timestamps

  field :nome, type: String
  field :sigla, type: String

  has_and_belongs_to_many :deputados


  def serializable_hash(options)
    super({ only: [:sigla]}.merge(options || {}))
  end

end
