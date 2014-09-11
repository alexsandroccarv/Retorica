class DiscursosController < ApplicationController
  before_action :set_discurso, only: [:show, :edit, :update, :destroy]

  # GET /discursos
  # GET /discursos.json
  def index
    @discursos = Discurso.all
  end

  # GET /discursos/1
  # GET /discursos/1.json
  def show
  end

  def import

    # Criando a conexão com o Servidor da Câmara

    data_inicial_legislatura = "01/02/2011".to_date
    data_final = Date.today


    # Calculando a diferença de dias entre a data de hoje e a data inicial da legislatura

    delta_days = data_final_legislatura - data_inicial

    # Calculando quantas iterações inteiras serão necessárias, pois o servidor da Câmara só aceita 360 dias

    integer_interation = delta_days.divmod(360)[0]

    # Calculando o resquício, caso não se tenha completado os 360 dias

    client = Savon.client(wsdl: 'http://www.camara.gov.br/SitCamaraWS/SessoesReunioes.asmx?wsdl')
    mod = delta_days.divmod(360)[1]




    puts discursos_list
  end



  private
    # Use callbacks to share common setup or constraints between actions.
    def set_discurso
      @discurso = Discurso.find(params[:id])
    end

    # Never trust parameters from the scary internet, only allow the white list through.
    def discurso_params
      params[:discurso]
    end
end
