from application.ports.command_publisher_port import CommandPublisherPort
from application.dtos.predicted_prices_dto import PredictedPricesDto
import pymongo

class CommandPublisherRepository(CommandPublisherPort):
  def __init__(self):
    mongoClient = pymongo.MongoClient("mongodb://localhost:27017/")
    self.db = mongoClient["stocks-prices-lstm"]

  def publish_historical_price(self, identifier: str, price: float, date_utc: str):
    self.db["HistoricalPrices"].insert_one({
      "identifier": identifier,
      "price": price,
      "date_utc": date_utc
    })

  def save_predicted_stock_prices(self, predicted_prices: list[PredictedPricesDto]):
    dict_list = list(map(
      lambda x: { "identifier": x.identifier, "price": x.price, "date_utc": x.date_utc }, predicted_prices
    ))
    print("save_predicted_stock_prices()")
    print(dict_list)

    self.db["PredictedPrices"].insert_many(dict_list)