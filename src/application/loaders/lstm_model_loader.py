from application.ports.query_model_port import QueryModelPort
from application.dtos.predicted_prices_dto import PredictedPricesDto
from domain.lstm_stocks_model import LstmStocksModel
import yfinance
import datetime
from collections import defaultdict

class LstmModelLoader:
  def __init__(self, query_model_port: QueryModelPort):
    self.query_model_port = query_model_port
    self.yfinance_data = {}
    self.predicted_prices = []

  def load_model(self):
    historical_prices = self.query_model_port.get_historical_prices()
    historical_identifiers_start_end = list(map(
      lambda hi: {
        'identifier': hi["identifier"],
        'price': hi["price"],
        'start_date_utc': hi["date_utc"],
        'end_date_utc': self.__get_end_date()
        },
      historical_prices
    ))
    
    grouped_historical_data = defaultdict(list)
    for hise in historical_identifiers_start_end:
      grouped_historical_data[hise["identifier"]].append(hise)
    
    for ghd in list(grouped_historical_data):
      min_value = min(
        ghd[1],
        key = lambda obj: obj["start_date_utc"]
      )
      self.__load_yahoo_finance_data(
        min_value["identifier"],
        min_value["start_date_utc"],
        min_value["end_date_utc"]
      )
    
    lstm_model = LstmStocksModel(historical_prices_dict=self.yfinance_data)
    lstm_model.train()

  def get_predicted_prices(self) -> list[PredictedPricesDto]:
    return self.predicted_prices

  def __get_end_date(self):
    now = datetime.datetime.now()

    return "{}-{}-{}".format(now.strftime("%Y"), now.strftime("%m"), now.strftime("%d"))

  def __load_yahoo_finance_data(self, sticker: str, start_date: str, end_date: str):
    self.yfinance_data[sticker] = yfinance(sticker, start=start_date, end=end_date)
