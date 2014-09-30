class Topic
  include Mongoid::Document

  field :title, type: String
  field :ignore, type: Mongoid::Boolean
  field :observ, type: Mongoid::Boolean

  belongs_to :dashboard

  has_many :emphases, :class_name => 'Emphasis'

  def serializable_hash(options)
    super({:include => [:emphases]}.merge(options))
  end

end
