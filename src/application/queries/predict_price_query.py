import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from application.ports.query_model_port import QueryModelPort
from application.queries.dtos.predicted_stock_price_dto import PredictedStockPriceDto

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

class PredictPriceQuery:
    def __init__(self, query_model_port: QueryModelPort):
        self.query_model_port = query_model_port
        # Carregar o modelo treinado LSTM
        self.model = tf.keras.models.load_model('lstm_stock_model.h5')
        # Configurar o escalador usado para normalizar os dados
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def get_stock_price(self, stock_identifier: str, date_utc: str) -> PredictedStockPriceDto:
        # Obter dados históricos da ação usando a porta de consulta
        historical_data = self.query_model_port.get_historical_stock_data(stock_identifier.strip().upper())

        if not historical_data:
            return None

        # Pré-processar os dados para o formato esperado pelo modelo LSTM
        data = np.array([x['close'] for x in historical_data])  # Supondo que 'close' contém os preços de fechamento
        data_scaled = self.scaler.fit_transform(data.reshape(-1, 1))

        # Criar a sequência de entrada para o modelo
        time_step = 60  # Número de passos temporais usados no treinamento
        if len(data_scaled) < time_step:
            raise ValueError("Dados insuficientes para prever o preço.")

        X_input = data_scaled[-time_step:].reshape(1, time_step, 1)

        # Previsão com o modelo LSTM
        predicted_price_scaled = self.model.predict(X_input)
        predicted_price = self.scaler.inverse_transform(predicted_price_scaled)[0][0]

        # Construir o DTO com o resultado da previsão
        return PredictedStockPriceDto(
            identifier=stock_identifier.strip().upper(),
            price=predicted_price,
            date_utc=date_utc.strip()
        )
