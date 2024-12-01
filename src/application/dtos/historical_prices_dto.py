import json

class HistoricalPricesDto:
  def __init__(self, identifier: str, price: float, date_utc: str):
    self.identifier = identifier
    self.price = price
    self.date_utc = date_utc

  def to_json(self):
    return json.dumps(self, default=lambda o: o.__dict__)