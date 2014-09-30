class Discurso
  include Mongoid::Document

  embeds_one :fase_sessao

  field :autor, type: String
  field :conteudo, type: String
  field :conteudo_plain_text, type: String
  field :conteudo_stemmed, type: String
  field :numero_insercao, type: Integer
  field :numero_orador, type: Integer
  field :proferido_em, type: DateTime
  field :sessao, type: String

end
