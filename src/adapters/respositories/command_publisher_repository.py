from application.ports.command_publisher_port import CommandPublisherPort
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