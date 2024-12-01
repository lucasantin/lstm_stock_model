from application.ports.command_publisher_port import CommandPublisherPort
from application.commands.dtos.save_stocks_data_dto import SaveStocksDataDto
from application.loaders.lstm_model_loader import LstmModelLoader

class SaveStocksDataCommand:
  def __init__(self, command_publisher_port: CommandPublisherPort, lstmLoader: LstmModelLoader):
    self.command_publisher_port = command_publisher_port
    self.lstmLoader = lstmLoader

  def handle(self, dto: SaveStocksDataDto):
    self.__validateDto(dto)

    self.command_publisher_port.publish_historical_price(
      dto.identifier.strip().upper(),
      dto.price,
      dto.date_utc.strip()
    )

    predicted_prices = self.lstmLoader.train_from_data()
    self.save_predicted_stock_prices(predicted_prices)

  def __validateDto(self, dto: SaveStocksDataDto):
    if dto == None or dto.price == None or len(dto.identifier) < 3 or len(dto.date_utc) != 10:
      raise Exception("Invalid input data at SaveStocksDataCommand.handle!")