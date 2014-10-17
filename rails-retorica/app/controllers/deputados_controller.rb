# -*- encoding : utf-8 -*-
class DeputadosController < ApplicationController
  before_action :set_deputado, only: [:show]

  # GET /deputados
  # GET /deputados.json
  def index
    Deputado.all
    respond_to do |format|
      format.json {}
    end
  end

  # GET /deputados/1
  # GET /deputados/1.json
  def show
    respond_to do |format|
      format.json{}
    end
  end

  private

  # Use callbacks to share common setup or constraints between actions.
  def set_deputado
    @deputado = Deputado.find(params[:id])
  end

  # Never trust parameters from the scary internet, only allow the white list through.
  def deputado_params
    params.require(:deputado).permit(:nome_parlamamentar, :partido, :sexo, :unidade_federativa, :email)
  end
end
