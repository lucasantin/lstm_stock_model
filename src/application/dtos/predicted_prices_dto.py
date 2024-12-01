class PredictedPricesDto:
  def __init__(self, identifier, price, date_utc):
    self.identifier = identifier
    self.price = price
    self.date_utc = date_utc