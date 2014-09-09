# -*- encoding : utf-8 -*-
class FiliacaoPartidaria
  include Mongoid::Document
  include Mongoid::Timestamps

  field :data_filiacao, type: Date
  belongs_to :deputado
  belongs_to :partido
end
