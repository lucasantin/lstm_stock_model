from model import train_and_predict

def make_prediction(symbol, start_date, end_date):
    try:
        current_price, prediction, mae, rmse, mape = train_and_predict(symbol, start_date, end_date)
        return {
            "current_price": float(current_price),
            "prediction": float(prediction),
            "mae": float(mae),
            "rmse": float(rmse),
            "mape": float(mape)
        }
    except ValueError as ve:
        return {"error": str(ve)}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}
