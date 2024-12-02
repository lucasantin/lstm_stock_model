import psutil
import time
from flask import Flask, request, jsonify
from predictor import make_prediction
import logging

app = Flask(__name__)

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Função para obter o uso de CPU e Memória
def get_system_usage():
    return {
        "cpu_usage": psutil.cpu_percent(interval=1),
        "memory_usage": psutil.virtual_memory().percent
    }

# Função para calcular variação do uso de recursos
def calculate_resource_variation(before, after):
    return {
        "cpu_usage_variation": after["cpu_usage"] - before["cpu_usage"],
        "memory_usage_variation": after["memory_usage"] - before["memory_usage"]
    }

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    required_params = ['symbol', 'start_date', 'end_date']
    if not data or not all(param in data for param in required_params):
        return jsonify({'error': f'Missing parameters. Required: {required_params}'}), 400

    symbol, start_date, end_date = data['symbol'], data['start_date'], data['end_date']
    logger.info(f"Received prediction request for {symbol} from {start_date} to {end_date}")

    try:
        start_time = time.time()
        system_usage_before = get_system_usage()

        result = make_prediction(symbol, start_date, end_date)

        elapsed_time = time.time() - start_time
        system_usage_after = get_system_usage()

        resource_variation = calculate_resource_variation(system_usage_before, system_usage_after)

        result.update({
            'response_time_seconds': float(elapsed_time),
            'resource_variation': resource_variation
        })

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return jsonify({'error': 'Error processing prediction'}), 500

if __name__ == '__main__':
    app.run(debug=True)
