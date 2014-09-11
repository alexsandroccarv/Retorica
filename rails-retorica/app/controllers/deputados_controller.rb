# -*- encoding : utf-8 -*-
class DeputadosController < ApplicationController
  before_action :set_deputado, only: [:show, :edit, :update, :destroy]

  # GET /deputados
  # GET /deputados.json
  def index
    @deputados = Deputado.all
  end

  # GET /deputados/1
  # GET /deputados/1.json
  def show
  end


  def import
    # Obtendo os deputados
    obter_deputados()

  end


  private
  # Use callbacks to share common setup or constraints between actions.
  def set_deputado
    @deputado = Deputado.find(params[:id])
  end

  # Never trust parameters from the scary internet, only allow the white list through.
  def deputado_params
    params.require(:deputado).permit(:nome, :partido, :sexo, :uf, :email)
  end

  def obter_deputados()

    # Por hora, vamos nos restringir apenas a última legislatura

    num_legislatura = 54

    # Criando a conexão com o Servidor da Câmara
    client = Savon.client(wsdl: 'http://www.camara.gov.br/SitCamaraWS/Deputados.asmx?wsdl')

    # Obtendo os deputados
    deputados_list = client.call(:obter_deputados).to_hash.to_json

    # Obtendo o array do ide_cadastros dos deputado para se consultar os dados completos
    ide_cadastros = JsonPath.on(deputados_list, '$..ide_cadastro')

    # Obtendo os cadastros completos de cada deputado
    ide_cadastros.each do |id|

    # Em um primeiro momento, vamos nos preocupar com os detalhes com o deputado da 54ª
    # legislatura, isso poderá ser melhorado no futuro
    deputado_response_details = client.call(:obter_detalhes_deputado,
            message: {ideCadastro: id, numLegislatura: num_legislatura}).to_hash.to_json

    # Caminho no arquivo JSON para localizar os dados de cada deputado
    deputado = JsonPath.on(deputado_response_details, '$..deputado')

    # Para cada deputado
    deputado.each do |item|

      #Obtém-se os dados Pessoais do Deputado
      nome_civil = item['nome_civil']
      nome_parlamentar_atual = item['nome_parlamentar_atual']
      sexo = item['sexo']
      data_nascimento = Date.parse(item['data_nascimento'])
      email = item['email']


      # Dados da Representação do Deputado
      unidade_federativa = UnidadeFederativa.find_or_create_by(:sigla => item['uf_representacao_atual'])
      situacao_legislatura = item['situacao_na_legislatura_atual']


      # Localizando o partido Atual do Deputado
      partido_atual = Partido.find_or_create_by(:nome => item.assoc('partido_atual')[1]['nome'],
                                             :sigla => item.assoc('partido_atual')[1]['sigla'])



      deputado_instance = Deputado.find_or_create_by(:_id => id, :nome_civil => nome_civil,
                                         :nome_parlamentar => nome_parlamentar_atual,
                      :sexo => sexo, :data_nascimento => data_nascimento,
                      :email => email,
                      :situacao_legislatura => situacao_legislatura,
                      :unidade_federativa => unidade_federativa)

      partido_atual.deputados << deputado_instance
      end
    end
  end
end
