import json
import numpy as np
import joblib
import os

def init():
    """
    Esta função é chamada quando o contêiner é iniciado. Ela é usada para carregar o modelo
    que será usado para inferência.
    """
    global model

    # Definir o caminho para o modelo. Usamos a variável de ambiente AZUREML_MODEL_DIR
    # para acessar o diretório onde o modelo foi carregado no contêiner.
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'anomaly_detection_model.pkl')
    
    # Verificar se o arquivo do modelo existe
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo não encontrado no caminho: {model_path}")

    # Carregar o modelo com joblib
    model = joblib.load(model_path)


def run(raw_data):
    """
    Esta função é chamada toda vez que uma requisição é enviada para o contêiner.
    Ela recebe os dados de entrada em formato JSON, processa-os, faz a previsão com o modelo 
    carregado e retorna os resultados como JSON.
    """
    try:
        # Analisar o JSON de entrada
        data = json.loads(raw_data)['input']
        
        # Converter os dados em uma matriz NumPy para realizar a previsão
        input_data = np.array(data)
        
        # Fazer as previsões usando o modelo carregado
        predictions = model.predict(input_data)
        
        # Gerar alertas para quaisquer anomalias detectadas (quando a previsão for -1)
        alerts = [f"Anomaly detected at row {i}" for i, pred in enumerate(predictions) if pred == -1]
        
        # Retornar as previsões e quaisquer alertas como um JSON
        return {"predictions": predictions.tolist(), "alerts": alerts}
    
    except Exception as e:
        # Se ocorrer algum erro durante a execução, retorne-o no formato JSON
        return {"error": str(e)}
