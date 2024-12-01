from application.dtos.predicted_prices_dto import PredictedPricesDto

class CommandPublisherPort:
  def __init__(self):
    pass

  def publish_historical_price(self, identifier: str, price: float, date_utc: str):
    pass

  def save_predicted_stock_prices(self, predicted_prices: list[PredictedPricesDto]):
    pass