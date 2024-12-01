from application.ports.dtos.stock_price_dto import StockPriceDto
from application.dtos.predicted_prices_dto import PredictedPricesDto
import json

class QueryModelPort:
  def get_predicted_stock_price(self, identifier: str, date_utc: str) -> StockPriceDto:
    pass

  def get_historical_prices(self):
    pass

  def save_predicted_stock_prices(self, predicted_prices: list[PredictedPricesDto]):
    pass

  def to_json(self):
    return json.dumps(self, default=lambda o: o.__dict__)