class DashboardsController < ApplicationController
  before_action :set_dashboard, only: [:show]
  respond_to :html, :json

  def index
    respond_with Dashboard.all
  end

  def show
    respond_with @dashboard
  end

  def first
    first_url = url_for Dashboard.first
    return redirect_to first_url
  end

  private

    def set_dashboard
      @dashboard = Dashboard.where(:slug => params[:id]).first
    end

end
