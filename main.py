import pandas as pd
import os
import datetime
import logging

# Cargamos funciones externas
from src.loader import cargar_datos #00
from src.features import feature_engineering_lag #01


# Cofiguracion de logging
os.makedirs("logs", exist_ok=True)

fecha = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
nombre_log = f"log_{fecha}.log"
logging.basicConfig(
                    level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(name)s %(lineno)d - %(message)s',
                    handlers=[logging.FileHandler(f"logs/{nombre_log}", mode='w', encoding='utf-8'),
                              logging.StreamHandler()]
                    )

logger = logging.getLogger(__name__)

# Funcion cargar datos


def main():
    logger.info("Inicio de ejecución del programa")
    
    #00 Cargar datos
    os.makedirs("data", exist_ok=True)
    path = 'data/competencia_01_crudo.csv'
    df = cargar_datos(path)
    
    
    #01 Feature engineering
    atributos = ["ctrx_quarter"]
    cant_lag = 2
    df = feature_engineering_lag(df, attributos=atributos, cant_lag=cant_lag)
      
    #02 Guardar datos
 
    logger.info(f"Fin de ejecución del programa,{nombre_log}")
    
   
if __name__ == "__main__":
    main()