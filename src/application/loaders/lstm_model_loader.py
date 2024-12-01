import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from application.dtos.predicted_prices_dto import PredictedPricesDto
from application.ports.query_model_port import QueryModelPort

class LstmModelLoader:
    def __init__(self, query_model_port: QueryModelPort):
        self.query_model_port = query_model_port
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def train_and_save_model(self):
        # Coletar preços históricos do banco de dados via QueryModelPort
        historical_prices = list(self.query_model_port.get_historical_prices())

        if not historical_prices:
            raise ValueError("Nenhum dado histórico encontrado para treinar o modelo.")

        # Extrair apenas os preços e ordená-los
        prices = [entry["price"] for entry in historical_prices]
        if len(prices) < 60:
            raise ValueError("Dados insuficientes para treinar o modelo LSTM. É necessário pelo menos 60 dias de preços.")

        # Pré-processamento dos dados
        data = np.array(prices).reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(data)

        # Criar sequências para treinamento
        X, y = self.__create_sequences(scaled_data, time_step=60)

        # Construir e treinar o modelo
        model = self.__build_lstm_model()
        model.fit(X, y, epochs=10, batch_size=64)

        # Salvar o modelo treinado
        model.save("lstm_stock_model.h5")
        print("Modelo LSTM treinado e salvo como 'lstm_stock_model.h5'")

    def __create_sequences(self, data, time_step=60):
        X, y = [], []
        for i in range(len(data) - time_step - 1):
            X.append(data[i:(i + time_step), 0])
            y.append(data[i + time_step, 0])
        return np.array(X).reshape(-1, time_step, 1), np.array(y)

    def __build_lstm_model(self):
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(60, 1)),
            LSTM(50, return_sequences=False),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        return model
