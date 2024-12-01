from application.ports.query_model_port import QueryModelPort
from application.ports.dtos.stock_price_dto import StockPriceDto
from application.dtos.predicted_prices_dto import PredictedPricesDto
import pymongo

class QueryModelRepository(QueryModelPort):
  def __init__(self):
    mongoClient = pymongo.MongoClient("mongodb://localhost:27017/")
    self.db = mongoClient["stocks-prices-lstm"]

  def get_predicted_stock_price(self, identifier: str, date_utc: str) -> StockPriceDto:
    query = {
      "identifier": identifier,
      "date_utc": date_utc
    }

    cursor = self.db["PredictedPrices"].find_one(query)

    if cursor == None:
      return cursor
    else:
      return StockPriceDto(cursor["identifier"], cursor["price"], cursor["date_utc"])

  def get_historical_prices(self):
    return self.db["HistoricalPrices"].find({ })
  
  def save_predicted_stock_prices(self, predicted_prices: list[PredictedPricesDto]):
    self.db["PredictedPrices"].insert_many(predicted_prices)