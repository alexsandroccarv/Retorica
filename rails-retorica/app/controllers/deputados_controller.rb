# -*- encoding : utf-8 -*-
class DeputadosController < ApplicationController
  before_action :set_deputado, only: [:show, :edit, :update, :destroy]

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
    params.require(:deputado).permit(:nome_parlamamentar, :partido, :sexo, :unidade_federativa, :email)
  end

  def obter_deputados()

    # Criando a conexão com o Servidor da Câmara
    client = Savon.client(wsdl: 'http://www.camara.gov.br/SitCamaraWS/Deputados.asmx?wsdl')

    # Obtendo os deputados
    deputados = JsonPath.on(client.call(:obter_deputados).to_hash.to_json, '$..deputado')

    # Para cada um dos deputados
    deputados[0].each do |deputado|

    # Lendo os parâmetros do JSON
    ide_cadastro = deputado['ide_cadastro']
    id_parlamentar = deputado['id_parlamentar']
    nome_parlamentar = deputado['nome_parlamentar']
    sexo = deputado['sexo']
    uf = deputado['uf']
    partido = deputado['partido']
    email = deputado['email']

    # Obtendo a URL da foto
    url_foto = deputado['url_foto']


    # Dados da Representação do Deputado
    unidade_federativa = UnidadeFederativa.find_or_create_by(:sigla => uf)
    partido_atual = Partido.find_or_create_by(:sigla => partido)


    deputado_instance = Deputado.find_or_create_by(:_id => ide_cadastro,
                                                   :site_deputado => " http://www.camara.leg.br/internet/deputado/Dep_Detalhe.asp?id=#{id_parlamentar}",
                                                   :nome_parlamentar => nome_parlamentar,
                                                   :sexo => sexo,
                                                   :email => email)

    deputado_instance.foto = URI.parse(url_foto)
    deputado_instance.save
    unidade_federativa.deputados << deputado_instance
    partido_atual.deputados << deputado_instance

    obter_enfase(deputado_instance,id_parlamentar)
    end
  end

  def obter_enfase(deputado_instance, id)

    csv = SmarterCSV.process(Rails.public_path+"autorFinal70.csv")
    authors_id = JsonPath.on(csv.to_json, '$..id')
    index = authors_id.index(id.to_i)

    unless index.nil?
      author_csv = csv[index]
      enfase = Enfase.find_or_create_by(:tema => author_csv[:rotulo], :valor => author_csv[:enfase])
      deputado_instance.enfases << enfase
      deputado_instance.save!
    end

  end

end