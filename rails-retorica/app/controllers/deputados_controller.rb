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

  # GET /deputados/new
  def new
    @deputado = Deputado.new
  end

  # GET /deputados/1/edit
  def edit
  end


  def import

    # Obtendo os deputados
    obter_deputados()

  end

  # POST /deputados
  # POST /deputados.json
  def create
    @deputado = Deputado.new(deputado_params)

    respond_to do |format|
      if @deputado.save
        format.html { redirect_to @deputado, notice: 'Deputado was successfully created.' }
        format.json { render action: 'show', status: :created, location: @deputado }
      else
        format.html { render action: 'new' }
        format.json { render json: @deputado.errors, status: :unprocessable_entity }
      end
    end
  end

  # PATCH/PUT /deputados/1
  # PATCH/PUT /deputados/1.json
  def update
    respond_to do |format|
      if @deputado.update(deputado_params)
        format.html { redirect_to @deputado, notice: 'Deputado was successfully updated.' }
        format.json { head :no_content }
      else
        format.html { render action: 'edit' }
        format.json { render json: @deputado.errors, status: :unprocessable_entity }
      end
    end
  end

  # DELETE /deputados/1
  # DELETE /deputados/1.json
  def destroy
    @deputado.destroy
    respond_to do |format|
      format.html { redirect_to deputados_url }
      format.json { head :no_content }
    end
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
      nome_civil = item['nome_civil'].capitalize.titleize
      nome_parlamentar_atual = item['nome_parlamentar_atual'].capitalize.titleize
      sexo = item['sexo']
      data_nascimento = Date.parse(item['data_nascimento'])
      email = item['email']


      # Dados da Representação do Deputado
      unidade_federativa = UnidadeFederativa.find_or_create_by(:sigla => item['uf_representacao_atual'])
      situacao_legislatura = item['situacao_na_legislatura_atual']


      # Localizando o partido Atual do Deputado
      partido_atual = Partido.find_or_create_by(:nome => item.assoc('partido_atual')[1]['nome'],
                                             :sigla => item.assoc('partido_atual')[1]['sigla'])

      # Criando a Filiação Partidária Atual
      filiacao_partidaria_atual = FiliacaoPartidaria.find_or_create_by(:partido => partido_atual)


      @filiacao_partidaria << filiacao_partidaria_atual


      filiacoes = item.assoc('filiacoesPartidarias')

      unless filiacoes.blank?
        filiacoes.reverse_each do |filiacao|

          # Nos dados providos pela Câmara, apenas a filiação posterior possui data de filiação

          partido_posterior = Partido.find_or_create_by(:sigla => filiacao['siglaPartidoPosterior'],
                                                        :nome => filiacao['nomePartidoPosterior'])

          data_filiacao_posterior = Date.parse(filiacao['dataFiliacaoPartidoPosterior'])

          filiacao_partidaria_posterior = FiliacaoPartidaria.find_or_create_by(:partido =>
                                                                                   partido_posterior, :data_filiacao => data_filiacao_posterior)

          # Adicionando a lista se não estiver lá
          @filiacao_partidaria << filiacao_partidaria_posterior unless
              @filiacao_partidaria.include?(filiacao_partidaria_posterior)


          partido_anterior = Partido.find_or_by(:sigla => filiacao['siglaPartidoAnterior'],
                                                :nome => filiacao['nomePartidoAnterior'])



          filiacao_partidaria_anterior = FiliacaoPartidaria.find_or_create_by(:partido => partido_anterior)


          # Adicionando a lista se não estiver lá
          @filiacao_partidaria << filiacao_partidaria_anterior unless
              @filiacao_partidaria.include?(filiacao_partidaria_anterior)

        end
      end


      puts @filiacao_partidaria

=begin
  deputado_instance = Deputado.find_or_create_by!(:_id => id, :nome_civil => nome_civil,
                                         :nome_parlamentar => nome_parlamentar_atual,
                      :sexo => sexo, :data_nascimento => data_nascimento,
                      :email => email,
                      :situacao_legislatura => situacao_legislatura, :partido => partido,
                      :unidade_federativa => unidade_federativa)
=end
      end
    end
  end
end
