# Projeto Identificação de Fraudes em cartões de crédito

Fonte dos dados: [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud)
É necessário fazer o download dos arquivos desse link acima, e extrair os dados do formato .zip para dentro da pasta 'dados' deste projeto a executar a análise exploratória e os demais notebooks .ipynb

O presente projeto tem como objetivo identificar as transações fraudulentas de um cartão de crédito. 

**ANÁLISE EXPLORATÓRIA**
Inicialmente foi feito a Análise Exploratória dos dados originais para conhecer a base, entender como é a distribuição dos dados de cada feature (utilizando para isso análises gráficas como Boxplots e Histplots), e fazer a análise de correlação entre as features. Uma vez que a base original tem grande volume de dados, foi feito o tratamento inicial dos dados de 2 maneiras:
- Remoção de Colunas: Feito teste estatístico de Mann_Whitneyu (para dados não paramétricos) a fim de identificar quais features poderiam ser removidas uma vez que, para um nível de significância de 99%, não influenciava a coluna alvo do projeto (Class)
- Remoção de Linhas: devido a grande presença de outliers em cada feature, foi removido os outliers de cada feature que represetavam os dados abaixo quartil inferior de 5% e superior ao quartil de 95%.
Essa base tratada foi exportada e foi usada para treinar os modelos de machine learning.

**PROJETO DE IDENTIFICAÇÃO DE FRAUDE** - foram feitos através de 3 notebooks:
1)  02.1_Modelo_LGBMClassifier_RandomUnderSampler
2)  02.2_Modelo_XGBClassifier_scale_pos_weight

**PRINCIPAIS CONSIDERAÇÕES / MÉTODOS UTILIZADOS NO PROJETO**

Mesmo com o **Tratamento dos Outliers** na análise exploratória os dados ainda apresentavam grandes quantidades de outliers, para isso foi usado o método "**RobustScaler**" nas etapas de preprocessamento do modelo.

Para **Balanceamento dos Dados** da coluna alvo (Class), no 1º notebook utilizamos o método **RandomUnderSampler** - que consiste em reduzir a quantidade de dados da classe majoritária para igualar com a minoritária. Já no 2º notebooks utilizamos parametros dentro dos modelos de classificação como "**class_weight='balanced**'" e "**scale_pos_weight**" a fim do próprio modelo ligar com o target desbalanceado. 

Em ambos os notebooks foram testados diversos **Modelos de Classificação** (LogisticRegression, SVC, KNeighboursClassifier, XGBClassifier, LGBMClassifier) a fim de analisar quais deles obteve melhor resultado e partir para otimização de parametros deste melhor modelo.

A **Principal Métrica** utilizada para avaliar os modelos foi "**average_precision**". Nessa aplicação é pior não detectar um frande de um cartão do que detectar uma transação verdadeira como fraude, pois podemos ter algum método de verificação com o cliente antes de alguma compra suspeita. Nesse sentido o recall é mais importante que a precisão, mas focar somente no recall pode trazer modelos que apontam tudo como fraude, o que é ruim também. Métricas como average_precision, que é a área de baixo da curva Precision-Recall, fazem um balanceamento entre os dados de recall e precisão e podem ser mais úteis nesse contexto.

Tendo o modelo otimizado foi utilizado métodos como **Plot Importance**e **Permutation Importance** (com parametros 'weight' e 'gain') para entender quais features tinham maior relevância para detecção das fraudes.

**PRINCIPAIS DIFERENÇAS E RESULTADOS ENTRE OS 3 NOTEBOOKS**
1)  **02.1_Modelo_LGBMClassifier_RandomUnderSampler**:
- Modelo de Classificação: LGBMClassifier da biblioteca LightGBM
- Balanceamento do target: usado método RandomUnderSampler (se usar o parametro scale_pos_weight os resultados serão inferiores)
- 'Average_precision" do modelo otimizado: 76,9%
- 'Recall' do modelo otimizado: 93%
- 'Precisão' do modelo otimizado: 60%
- Principais features da Plot Importance: V4, V14, V12, V10, Amount

**OBS**: Como o desbalanceamento de classes é muito grande (fraudes não representavam nem 1% das transações), ao utilizar o método RandomUnderSampler reduzimos demais os dados, podendo gerando o under fitting dos modelos treinados após isso (por conta disso a precisão pode ter sido menor no 1º notebook). Poderíamos testar algum método de balanceamento de classes "híbrido" (que combina under e over sampling) para ver se conseguimos um resultado do average_precision e da precisão um pouco maior sem comprometer o recall. 

2)  **02.2_Modelo_XGBClassifier_scale_pos_weight**: 
- Modelo de Classificação: XGBClassifier da biblioteca XGBoost
- Balanceamento do target: usado parametros scale_pos_weight dentro do modelo, não usado nenhum método imbalance_learn
- Redução das features - Permutation Importance: já que não foi utilizado o método RandomUnderSampler como no primeiro notebook, tinhamos uma base muito maior, o que acarretaria em muito mais tempo para buscar a otimização dos parametros pelo GridSearchCV. Por conta disso, foi utilizado o Permutation_Importance para entender quais features quando embaralhadas geravam menor queda de desempenho do modelo de classificação. Aquelas features cuja importância pro modelo era menor do que 1% foram eliminadas da base que foi utilizada pelo GridSearch para buscar o modelo otimizado do LGBMClassifier.
- 'Average_precision" do modelo otimizado: 87%
- 'Recall' do modelo otimizado: 83%
- 'Precisão' do modelo otimizado: 89%
- 'Average_precision" do modelo padrão XGBClassifier: 88%
- 'Recall' do modelo padrão XGBClassifier: 84%
- 'Precisão' do modelo padrão XGBClassifier: 94%
- Principais features da Plot Importance:  V4, V14 e Amount

**OBS**: Na busca pela otimização dos parametros do 2º notebook, utilizando o GridSearchCV com métrica average_precision utilizada como refit, não tivemos ganhos significativos com a otimização dos parametros e a precisão reduziu 5% ainda. Isso aconteceu devido a reduçao de features para ganhar tempo ao rodar o grid_search. Nesse caso é melhor usar o modelo padrão do XGBClassifier com todas as features.

**CONSIDERAÇÕES FINAIS**
É fácil notar que os resultados do segundo notebook foram muito superiores ao primeiro notebook olhando pro average_precision (11% superior), mas analisando pelo recall o segundo notebook teve resultados quase 10% menores (93% contra 84%). Comparando a precisão de ambos os modelos tivemos um ampla vantagem novamente pro segundo notebook (94% contra 60% do primeiro notebook).

A escolha do melhor modelo depende dos custos envolvendo a não detecção de fraudes (falso positivo para fraude) e da detecção excessiva de transações verdadeiras como fraudulentas (falso positivo para fraude). 

Para todos os notebooks podemos concluir que as features V4, V14 e Amount são as que tem maior influência na detecção das fraudes, e podem ser usadas ou exploradas mais pelos bancos para evitar ou conferir com seus clientes transações suspeitas.

## Organização do projeto

```
├── .env               <- Arquivo de variáveis de ambiente (não versionar)
├── .gitignore         <- Arquivos e diretórios a serem ignorados pelo Git
├── ambiente.yml       <- O arquivo de requisitos para reproduzir o ambiente de análise
├── LICENSE            <- Licença de código aberto se uma for escolhida
├── README.md          <- README principal para desenvolvedores que usam este projeto.
|
├── dados              <- Arquivos de dados para o projeto.
|
├── modelos            <- Modelos treinados e serializados, previsões de modelos ou resumos de modelos
|
├── notebooks          <- Cadernos Jupyter. A convenção de nomenclatura é um número (para ordenação),
│                         as iniciais do criador e uma descrição curta separada por `-`, por exemplo
│                         `01-fb-exploracao-inicial-de-dados`.
│
|   └──src             <- Código-fonte para uso neste projeto.
|      │
|      ├── __init__.py  <- Torna um módulo Python
|      ├── config.py    <- Configurações básicas do projeto
|      └── graficos.py  <- Scripts para criar visualizações exploratórias e orientadas a resultados
|
├── referencias        <- Dicionários de dados, manuais e todos os outros materiais explicativos.
|
├── relatorios         <- Análises geradas em HTML, PDF, LaTeX, etc.
│   └── imagens        <- Gráficos e figuras gerados para serem usados em relatórios
```
