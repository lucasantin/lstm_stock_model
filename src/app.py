from flask import Flask, request
from application.commands.save_stocks_data_command import SaveStocksDataCommand
from application.commands.dtos import save_stocks_data_dto
from application.queries.predict_price_query import PredictPriceQuery
from application.loaders.lstm_model_loader import LstmModelLoader
from adapters.respositories.query_model_repository import QueryModelRepository
from adapters.respositories.command_publisher_repository import CommandPublisherRepository

app = Flask(__name__)

base_route = "/stocks-guru"

save_stocks_command = SaveStocksDataCommand(
  CommandPublisherRepository(),
  LstmModelLoader(QueryModelRepository())
)
predict_price_query = PredictPriceQuery(QueryModelRepository())


@app.route(base_route + "/login", methods=["POST"])
def post_login():
  print("Request Data:")
  print(request.json.get('password'))
  print(request.json.get('username'))

  return { "jwt": "" }

@app.route(base_route + "/stocks/data", methods=["POST"])
def post_save_stocks_data():
  save_stocks_command.handle(save_stocks_data_dto.SaveStocksDataDto(
    request.json.get("identifier"),
    request.json.get("price"),
    request.json.get("utc_date_time")
  ))

  return { "success": True }

@app.route(base_route + "/stocks/<identifier>", methods=["GET"])
def get_predict_stocks_price(identifier: str):
  print("Querying data for {} at {}".format(identifier, request.args.get('date', '')))
  response_data = predict_price_query.get_stock_price(identifier, request.args.get('date', ''))
  
  if response_data == None:
    return { "data": None }
  
  return { "data": response_data.to_json() }
