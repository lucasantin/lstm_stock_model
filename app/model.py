import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
import math

# Função para preparar os dados com indicadores técnicos
def prepare_data(symbol, start_date, end_date):
    # Baixar dados históricos do Yahoo Finance
    data = yf.download(symbol, start=start_date, end=end_date)

    # Criar indicadores técnicos
    data['SMA_20'] = data['Close'].rolling(window=20).mean()  # Média móvel de 20 dias
    data['SMA_50'] = data['Close'].rolling(window=50).mean()  # Média móvel de 50 dias
    data['RSI'] = calculate_rsi(data['Close'])  # Índice de Força Relativa (RSI)
    data['Volume'] = data['Volume']  # Volume de negociações

    # Preencher valores NaN gerados pelos indicadores
    data = data.dropna()

    # Usar apenas colunas relevantes
    features = ['Close', 'SMA_20', 'SMA_50', 'RSI', 'Volume']
    data = data[features]

    # Normalizar os dados para uso no LSTM
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Dividir os dados em treino e teste (80% treino e 20% teste)
    train_size = int(len(scaled_data) * 0.8)
    train, test = scaled_data[:train_size], scaled_data[train_size:]

    # Converter dados em formato apropriado para o LSTM
    def create_dataset(dataset, time_step=1):
        X, Y = [], []
        for i in range(len(dataset) - time_step):
            X.append(dataset[i:i + time_step])
            Y.append(dataset[i + time_step, 0])  # Prever o preço de fechamento
        return np.array(X), np.array(Y)

    time_step = 60  # Janela temporal de 60 dias
    X_train, Y_train = create_dataset(train, time_step)
    X_test, Y_test = create_dataset(test, time_step)

    return X_train, Y_train, X_test, Y_test, scaler, data

# Função para calcular o RSI
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(0)

# Função para construir o modelo LSTM
def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=64, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Função para treinar o modelo e fazer previsões
def train_and_predict(symbol, start_date, end_date):
    X_train, Y_train, X_test, Y_test, scaler, data = prepare_data(symbol, start_date, end_date)
    
    model = build_model((X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, Y_train, epochs=50, batch_size=32, verbose=1)

    # Fazer previsões para o último valor do conjunto de teste
    last_sequence = X_test[-1].reshape(1, X_test.shape[1], X_test.shape[2])
    prediction = model.predict(last_sequence)[0][0]

    # Desnormalizar os valores
    prediction = scaler.inverse_transform([[prediction] + [0] * (len(data.columns) - 1)])[0][0]
    current_price = data.iloc[-1]['Close']

    # Calcular métricas
    mae = mean_absolute_error(Y_test, model.predict(X_test).flatten())
    rmse = math.sqrt(mean_squared_error(Y_test, model.predict(X_test).flatten()))
    mape = np.mean(np.abs((Y_test - model.predict(X_test).flatten()) / Y_test)) * 100

    return current_price, prediction, mae, rmse, mape
