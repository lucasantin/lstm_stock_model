# Previsão de Preços de Ações - API de Previsão com Flask

Este projeto fornece uma **API de Previsão de Preços de Ações** utilizando um modelo de **Deep Learning (LSTM)**, que é uma rede neural especializada em sequências de dados, como séries temporais. O modelo é treinado para prever o preço futuro de uma ação com base em dados históricos. A API é construída com **Flask**, um microframework para Python que facilita a criação de APIs simples e eficientes. Além disso, o sistema inclui funcionalidades para monitorar o desempenho do sistema, como uso de CPU e memória, e calcula métricas de erro para avaliar a precisão das previsões feitas pelo modelo.

## Estrutura do Projeto

O projeto é composto por três arquivos principais, responsáveis por diferentes etapas do processo de previsão de preços das ações:

1. **`app.py`**: Contém a configuração da API Flask, que lida com as requisições HTTP e responde com as previsões.
2. **`predictor.py`**: Orquestra o processo de previsão, incluindo o carregamento do modelo treinado e a execução da previsão.
3. **`model.py`**: Contém a lógica de construção e treinamento do modelo LSTM, além da preparação dos dados e o cálculo das métricas de erro.

### Descrição dos Arquivos

#### 1. `app.py` - A API Flask

Este arquivo define a estrutura da API que processa as requisições HTTP e realiza as previsões. A principal rota da API é a `/predict`, que aceita requisições `POST` contendo os parâmetros necessários para a previsão.

##### Funcionalidade:

- **Recebe os dados da requisição**: A API recebe os parâmetros necessários, como o símbolo da ação (`symbol`), a data de início (`start_date`) e a data de fim (`end_date`).
- **Verifica os parâmetros**: Caso algum parâmetro esteja faltando ou seja inválido, a API retorna um erro com a descrição do problema.
- **Monitora o uso de recursos do sistema**: Antes e depois da execução da previsão, a API verifica o uso de CPU e memória do sistema, para avaliar o impacto da execução no desempenho do servidor.
- **Chama a função `make_prediction`**: A função presente no arquivo `predictor.py` é chamada para realizar a previsão do preço da ação com base nos dados fornecidos.
- **Retorna os resultados**: A resposta inclui o preço atual da ação, a previsão do preço futuro, as métricas de erro (MAE, RMSE, MAPE), o tempo de resposta da previsão e a variação do uso de CPU e memória.

##### Exemplo de Endpoint:

Requisição POST para `http://127.0.0.1:5000/predict`:

json

    "symbol": "AAPL",
    "start_date": "2020-01-01",
    "end_date": "2024-11-20"


##### Exemplo de resposta JSON:

{
  "current_price": 228.27999877929688,
  "mae": 0.03235111360211602,
  "mape": 3.727798039969654,
  "prediction": 219.03649022883474,
  "resource_variation": {
    "cpu_usage_variation": -22.5,
    "memory_usage_variation": -1.2000000000000028
  },
  "response_time_seconds": 60.34456825256348,
  "rmse": 0.040244925627786456
}

#### 2. predictor.py - Orquestrador da Previsão
O arquivo predictor.py é o responsável por orquestrar a execução do processo de previsão. Ele chama o modelo treinado e lida com os parâmetros de entrada, retornando os resultados da previsão.

##### Funcionalidade:
Recebe os parâmetros da requisição: O arquivo extrai os parâmetros de entrada (símbolo da ação, data de início e data de fim) da requisição recebida.
Chama o modelo para fazer a previsão: A função load_and_predict() presente no arquivo model.py é chamada para carregar o modelo previamente treinado e fazer a previsão de preço da ação.
Retorna os resultados: O arquivo retorna a previsão, as métricas de erro e as informações de uso de recursos, como CPU e memória.

#### 3. model.py - Modelo LSTM
Este arquivo contém a lógica responsável pela construção do modelo de previsão, preparação dos dados e cálculo das métricas de erro. O modelo LSTM (Long Short-Term Memory) é uma rede neural especializada em sequências temporais, como séries de preços de ações.

##### Funcionalidade:
Prepara os dados: Utiliza a biblioteca yfinance para baixar os dados históricos do preço das ações, faz a normalização dos dados e os divide em conjuntos de treino e teste.
Carrega o modelo treinado: Carrega o modelo de previsão que foi treinado previamente e salvo em um arquivo .h5.
Faz a previsão: Utiliza o modelo carregado para prever o preço da ação com base nos dados históricos fornecidos.
Calcula as métricas de erro: Calcula três métricas de erro para avaliar a precisão da previsão:
**MAE (Mean Absolute Error)**: Mede a diferença média entre o valor real e o valor previsto.
**RMSE (Root Mean Squared Error)**: Mede a raiz quadrada da média dos erros quadráticos.
**MAPE (Mean Absolute Percentage Error)**: Mede o erro percentual médio entre os valores reais e os previstos.

## Requisitos
Antes de rodar o projeto, é necessário instalar algumas dependências. Para isso, basta utilizar o arquivo requirements.txt, que contém todas as bibliotecas necessárias:
O projeto depende das seguintes bibliotecas:

**Flask: **Framework para criar APIs web.
**yfinance: **Para obter dados financeiros históricos de ações.
**TensorFlow: **Biblioteca para construir e treinar redes neurais.
**scikit-learn: **Para calcular métricas de erro como MAE, RMSE e MAPE.
**psutil: **Para monitorar o uso de CPU e memória.
**numpy: **Para manipulação de dados numéricos.
**pandas: **Para manipulação de dados estruturados.

## Como Funciona
A API realiza previsões de preços de ações com base em dados históricos. O fluxo de trabalho pode ser dividido em duas fases principais: Treinamento e Inferência.

### 1. Treinamento
O modelo LSTM (Long Short-Term Memory) é treinado previamente com dados históricos de ações. O treinamento envolve os seguintes passos:

Preparação dos Dados: Baixam-se os dados históricos de preços de fechamento das ações usando a API yfinance. Esses dados são normalizados (ajustados para uma faixa entre 0 e 1) e divididos em conjuntos de treino e teste.
Construção do Modelo LSTM: O modelo LSTM é construído utilizando duas camadas LSTM, seguidas por uma camada densa (fully connected).
Treinamento do Modelo: O modelo é treinado utilizando os dados de treino. Durante o processo de treinamento, as métricas de erro (MAE, RMSE, MAPE) são calculadas para avaliar o desempenho do modelo.
Salvamento do Modelo: Após o treinamento, o modelo é salvo em um arquivo .h5 para ser carregado posteriormente durante a inferência.

### 2. Inferência
Quando a API recebe uma requisição para fazer uma previsão, o processo de inferência ocorre. O modelo treinado é carregado e utilizado para prever o preço de fechamento de uma ação com base nos dados fornecidos.

#### Etapas da Inferência:
**Recebimento de Dados: **A API recebe os parâmetros da requisição: o símbolo da ação (symbol), a data de início (start_date) e a data de fim (end_date).
**Carregamento do Modelo: **O modelo previamente treinado e salvo em arquivo .h5 é carregado.
**Previsão: **Com os dados fornecidos, o modelo faz a previsão do preço da ação.
**Cálculo das Métricas de Erro: **MAE, RMSE e MAPE são calculados para avaliar a precisão da previsão feita pelo modelo.
**Monitoramento de Recursos: **A API monitora o uso de CPU e memória antes e depois da execução da previsão.
**Resposta: **A API retorna a previsão, as métricas de erro, o tempo de resposta e as informações sobre a variação de uso de recursos.

## Monitoramento de Recursos
Durante a execução da previsão, a API também realiza o monitoramento de recursos do sistema. Isso inclui a verificação do uso de CPU e memória antes e depois da execução do modelo. O objetivo é entender o impacto da execução do modelo no desempenho do sistema.

#### Exemplo de Variação de Recursos:
"resource_variation": {
  "cpu_usage_variation": -22.5,
  "memory_usage_variation": -1.2000000000000028
}

## Métricas de Erro
As métricas de erro são essenciais para avaliar a precisão do modelo de previsão. O projeto calcula três métricas principais:

**MAE (Erro Absoluto Médio): **Mede a diferença média entre os valores reais e os valores previstos. Quanto menor o MAE, melhor a precisão do modelo.
**RMSE (Raiz do Erro Quadrático Médio): **Mede a raiz quadrada da média dos erros quadráticos. Essa métrica é sensível a grandes erros, sendo útil quando se deseja penalizar previsões grandes imprecisas.
**MAPE (Erro Percentual Absoluto Médio): **Mede o erro percentual médio entre os valores reais e os valores previstos. Ele é útil para avaliar o erro relativo, mas pode ser distorcido se o valor real for muito pequeno.