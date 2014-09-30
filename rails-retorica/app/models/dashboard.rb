class Dashboard
  include Mongoid::Document

  field :title, type: String
  field :slug, type: String

  has_many :topics

  def serializable_hash(options)
    super({:include => [:topics]}.merge(options))
  end
end
