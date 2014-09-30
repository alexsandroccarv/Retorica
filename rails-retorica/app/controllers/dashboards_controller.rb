class DashboardsController < ApplicationController
  before_action :set_dashboard, only: [:show]
  respond_to :html, :json

  def index
    respond_with Dashboard.all
  end

  def show
    respond_with @dashboard
  end

  private

    def set_dashboard
      @dashboard = Dashboard.where(:slug => params[:id]).first
    end

end
