from application.ports.dtos.stock_price_dto import StockPriceDto
from application.dtos.historical_prices_dto import HistoricalPricesDto
import json

class QueryModelPort:
  def get_predicted_stock_price(self, identifier: str, date_utc: str) -> StockPriceDto:
    pass

  def get_historical_prices(self) -> list[HistoricalPricesDto]:
    pass

  def to_json(self):
    return json.dumps(self, default=lambda o: o.__dict__)