# -*- encoding : utf-8 -*-
class DeputadosController < ApplicationController
  before_action :set_deputado, only: [:show, :edit, :update, :destroy]

  NUM_LEGISLATURA = '54'

  # GET /deputados
  # GET /deputados.json
  def index

   @deputados = []
  # Colocando Somente os deputados com discursos
   Deputado.all.each do |deputado|

    if deputado.enfases.exists?
    @deputados << deputado
    end

   end

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

  def discursos
  end



  def import
    obter_deputados_arquivo
    respond_to do |format|
      format.html{

        #redirect_to root_path, notice: 'Deputados atualizados com sucesso.'
        }
    end

    #obter_deputados_webservice()

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


  def obter_deputados_arquivo
    url = "http://www.camara.leg.br/internet/Deputado/DeputadosXML.zip"

    # Salvando o arquivo dos deputados
    open('public/deputados.zip', 'wb') do |file|
      file << open(url).read

      # Extraindo os arquivos

    Zip::File.open(file) do |zip_file|
        zip_file.each do |entry|
          entry.extract("public/deputados.xml") unless File.exists?("public/deputados.xml")
        end
      end
    end

    # Abrindo arquivo Hash
    # Obtendo os deputados da 54ª legislatura (precisa ser melhorado depois)
    hash = Hash.from_xml(File.open('public/deputados.xml', "r:UTF-8"))

    # Navegando na estrutura
    deputados_hash = hash['orgao']['Deputados']['Deputado']

    # Array onde será armazenado o ide_cadastro de cada deputado
    @deputados_ids = []
    deputados_hash.each do |deputado|
    if deputado['numLegislatura'] == NUM_LEGISLATURA
      obter_detalhes_deputado(deputado["ideCadastro"])
    end
    end
  end

  def obter_detalhes_deputado(ide_cadastro)
  # Criando a conexão com o Servidor da Câmara
    client = Savon.client(wsdl: 'http://www.camara.gov.br/SitCamaraWS/Deputados.asmx?wsdl')

    deputado_response_details = client.call(:obter_detalhes_deputado,
                                            message: {ideCadastro: ide_cadastro, numLegislatura: NUM_LEGISLATURA}).to_hash.to_json


    deputado = JsonPath.on(deputado_response_details, '$..deputado')

    # Para cada deputado
    deputado.each do |item|

      #Obtém-se os dados Pessoais do Deputado
      nome_parlamentar = item['nome_parlamentar_atual']
      sexo = item['sexo']
      email = item['email']

      # Dados da Representação do Deputado
      unidade_federativa = UnidadeFederativa.find_or_create_by(:sigla => item['uf_representacao_atual'])
      situacao = item['situacao_na_legislatura_atual']


      # Localizando o partido Atual do Deputado
      partido_atual = Partido.find_or_create_by(:nome => item.assoc('partido_atual')[1]['nome'],
                                                :sigla => item.assoc('partido_atual')[1]['sigla'])


      deputado_instance = Deputado.find_or_create_by(:_id => ide_cadastro,
                                                     :site_deputado => " http://www.camara.leg.br/internet/deputado/Dep_Detalhe.asp?id=#{ide_cadastro}",
                                                     :nome_parlamentar => nome_parlamentar,
                                                     :sexo => sexo,
                                                     :email => email, :situacao => situacao)



      url_foto = "http://www.camara.gov.br/internet/deputado/bandep/#{ide_cadastro}.jpg"

      deputado_instance.foto = URI.parse(url_foto)
      deputado_instance.save
      unidade_federativa.deputados << deputado_instance
      partido_atual.deputados << deputado_instance

      obter_enfase(deputado_instance,nome_parlamentar)

    end
  end

  def obter_enfase(deputado_instance, nome_parlamentar)

    csv = SmarterCSV.process(Rails.public_path+"autorFinal70.csv") if csv.nil?

    csv.each do  |csv_element|
      autor = csv_element[:autor]
      #if nome_parlamentar.downcase =~ /^abelar/
      #  require 'byebug'; byebug
      #end
      if autor ==  I18n.transliterate(nome_parlamentar)
        enfase = Enfase.find_or_create_by(:tema => csv_element[:rotulo], :valor => csv_element[:enfase])
        deputado_instance.enfases << enfase
        deputado_instance.save!

      end

    end
  end
end
