class Emphasis
  include Mongoid::Document

  field :name, type: String
  field :emphasis, type: Float

  belongs_to :topic
  belongs_to :deputado

  def serializable_hash(options)
    super({:include => [:deputado]}.merge(options))
  end
end
