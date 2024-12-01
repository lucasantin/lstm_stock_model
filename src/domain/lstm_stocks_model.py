from collections import defaultdict

class LstmStocksModel:
  def __init__(self, historical_prices_dict: dict):
    self.historical_prices_dict = historical_prices_dict
    self.predicted_prices = defaultdict(list)

  def train(self):
    ## Implement training code here
    pass

  def get_predicted_prices(self):
    return self.predicted_prices
