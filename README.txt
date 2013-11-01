Documentação do Projeto Retórica

Retórica é uma aplicação web que permite a todos os cidadãos conhecerem os temas mais enfatizados pelos Deputados Federais em seus discursos. 

O projeto utiliza os discursos proferidos no Pequeno Expediente pelos Deputados federais da legislatura 2011-2014. Nesta análise foram considerados os discursos proferidos entre o começo da 54a Legislatura (02/02/2011) e primeiro de outubro de 2013 (inclusive). Apenas Deputados com mais de um discurso proferido nesse período foram incluídos na análise.

Nessa aplicação é possível descobrir quais os Deputados Federais que enfatizam os mesmos temas ao discursar. 

Motivação

O processo legislativo brasileiro e a atividade parlamentar não consistem apenas em votos e na tramitação de matérias. A política brasileira, sob o sistema democrático de representação, também é desenvolvida a partir do diálogo, do debate, da exposição de idéias e de preferências. Por esta razão a Constituição de 1988 e o Regimento Interno da Câmara dos Deputados garantem aos parlamentares o direito à palavra através de espaços institucionais para seu pronunciamento como, por exemplo, o Grande e o Pequeno expedientes.

Após inúmeras pesquisas entre estudiosos de ciência política, sabe-se que as votações nominais realizadas ao longo dos processos legislativos são partidariamente organizadas, reduzindo a expressão individual de cada parlamentar. Logo, é justamente nos espaços reservados para seu pronunciamento que o deputado pode entrar em contato com seus eleitores e assim divulgar suas idéias e pensamentos de forma mais livre. Com acesso fácil e ágil aos temas proferidos por cada deputado, os cidadãos brasileiros podem encontrar afinidades políticas e de ideias com seus representantes. 

Metodologia

Para a identificação de temas presentes nos pronunciamentos de cada parlamentar, o projeto se baseia na metodologia desenvolvida pelo Professor Justin Grimmer da Universidade de Stanford, conhecida como expressed agenda model. Para tanto, foram necessárias quatro etapas:

1) Obtenção dos dados:

Infelizmente os dados dos pronunciamentos ainda não estão à disposição no portal de dados abertos (http://www2.camara.leg.br/transparencia/dados-abertos). Por esta razão, desenvolvemos um web-crawler para obter os dados dos pronunciamentos presentes na página da Câmara dos Deputados (www.camara.gov.br). Foram coletados todos os pronunciamentos do pequeno expediente proferidos entre 02/02/2011 e 01/11/2013.

2) Limpeza e organização dos dados

Em seguida, cada pronunciamento passou por uma "limpeza" com o objetivo de se obter uma “sacola de palavras” (bag of words) de cada um. Esta “sacola de palavras” permite identificar a frequência com que cada palavra ocorre num pronunciamento, de forma que se possa agrupar os pronunciamentos em temas comuns.
Para que se possa aplicar o algoritmo, é necessário garantir que palavras que variam apenas na flexão, número, conjugação etc. (p. ex., protestaram e prostou-se, ou árvore e árvores) sejam consideradas as mesmas palavras. Assim, a “limpeza” dos pronunciamentos consiste nos seguintes passos:

Em primeiro lugar, todos os deputados com apenas um pronunciamento (ou sem pronunciamentos) foram removidos da Análise. Assim, os discursos desses deputados não foram considerados. Em segundo lugar,  todas as palavras foram colocadas em caixa baixa e todos os acentos e caracteres de pontuação foram removidos Também foram removidas palavras muito utilizadas e palavras pouco utilizadas. Por fim, através da adaptação do algorítimo de Porter (1980) para o português já desenvolvido por diferentes projetos (Snowball e NILC-USP), foram obtidas as raízes (stems) das palavras restantes.

Ao final, foi construída uma matriz (Document Term Matrix), cujas linhas representam cada pronunciamento e cujas palavras representam as colunas. É esta matriz que foi utilizada na estimação dos tópicos das falas dos deputados. Nessa matriz, restaram 447 deputados e 14.500 documentos.

3) Estimação de tópicos

Com base na metodologia de modelagem de tópicos (topic model) do expressed agenda model. Foram estimados os tópicos presentes em cada pronunciamento feito pelos parlamentares.

De forma simplificada, esta metodologia permite agrupar discursos similares em temas comuns de acordo com seus autores. Quanto mais similares forem diferentes discursos (em termos da frequência com que cada palavra ocorre em cada um), maior a chance deles serem agrupados sob o mesmo tema. Se houver pouca similaridade nos temas dos discursos, o algoritmo agrupará os discursos em temas diferentes.

Essa técnica está presente no campo de aprendizagem computacional não supervisionada (unsupervised machine learning),  de modo que ela dispensa a definição de temas a priori por um analista e, portanto, torna o processo de classificação inteiramente automatizado. Os temas sob os quais os discursos serão agrupados serão definidos a partir dos dados obtidos nos própios discursos dos parlamentares.

Sendo um modelo estatístico hierárquico Bayesiano de modelagem de tópicos, o expressed agenda model permite descobrir os temas mais enfatizados pelos diferentes parlamentares. Leva-se em consideração o fato de que as palavras utilizadas por um Deputado Federal em cada discurso não são aleatórias, mas obedecem à estrutura linguística discursiva de cada deputado. Assim, conseguimos captar a ênfase temática dos discursos de cada um dos 447 deputados analisados. Este algoritmo foi implementado em R, conforme descrito em artigo de Grimmer (2010) [1].


Visualização

No endereço http://luiscarli.com/retorica , é possível encontrar uma visualização interativa dos temas enfatizados pelos Deputados no Pequeno Expediente. O usuário tem a sua disposição um bubble chart (Gráfico de bolhas), em que cada bolha representa a relevância (medido pela frequência) de cada tema dentre todos os 447 deputados analisados. Dentro de cada bolha temos os deputados que enfatizam aquele tema nos seus discursos. Na visualização, um deputado está associado a um único tema, que é o tema mais enfatizado por cada deputado nos discursos por eles proferidos. Clicando num deputado, é possível visualizar mais informações sobre ele (e-mail, link pra url do deputado etc.).




[1] Grimmer, Justin. "A Bayesian hierarchical topic model for political texts: Measuring expressed agendas in Senate press releases." Political Analysis18.1 (2010): 1-35.
