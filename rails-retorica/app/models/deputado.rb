# -*- encoding : utf-8 -*-
class Deputado
  include Mongoid::Document
  include Mongoid::Attributes::Dynamic
  include Mongoid::Timestamps
  include Mongoid::Paperclip

  has_mongoid_attached_file :foto

  field :ide_cadastro, type: String
  field :situacao, type: String
  field :site_deputado, type: String
  field :foto_url, type: String
  field :nome_parlamentar, type: String
  field :sexo, type: String
  field :email, type: String

  has_and_belongs_to_many :partidos
  has_and_belongs_to_many :unidade_federativa

  has_many :emphases, :class_name => 'Emphasis'

  validates_attachment :foto, content_type: { content_type: ["image/jpg", "image/jpeg", "image/png", "image/gif"] }

  def serializable_hash(options)
    super({
      include: [:partidos, :unidade_federativa, :foto_url]
    }.merge(options || {}))
  end

  def foto_url
    if foto.exists?
      foto.url
    else
      ActionController::Base.helpers.asset_path 'null.jpeg'
    end
  end
end
