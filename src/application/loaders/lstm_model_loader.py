from application.ports.query_model_port import QueryModelPort
from application.dtos.predicted_prices_dto import PredictedPricesDto

class LstmModelLoader:
  def __init__(self, query_model_port: QueryModelPort):
    self.query_model_port = query_model_port
    self.yfinabnce_data = None

  def train_from_data(self) -> list[PredictedPricesDto]:
    historical_prices = self.query_model_port.get_historical_prices()
    self.__load_yahoo_finance_data()

    return historical_prices

  def __load_yahoo_finance_data(self):
    print("Loading data...")
    self.yfinabnce_data = {}
