import logging
from datetime import datetime
import os
import pandas as pd

from src.features import feature_engineering_lag
from src.loader import cargar_datos, convertir_clase_ternaria_a_target
from src.best_params import cargar_mejores_hiperparametros
from src.final_training import preparar_datos_entrenamiento_final, generar_predicciones_finales, entrenar_modelo_final
from src.output_manager import guardar_predicciones_finales
from src.optimization import optimizar, evaluar_en_test, guardar_resultados_test
from src.best_params import obtener_estadisticas_optuna
from src.conf import *


# Cofiguracion de logging
os.makedirs("logs", exist_ok=True)

fecha = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
nombre_log = f"log_{fecha}.log"
logging.basicConfig(
                    level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(name)s %(lineno)d - %(message)s',
                    handlers=[logging.FileHandler(f"logs/{nombre_log}", mode='w', encoding='utf-8'),
                              logging.StreamHandler()]
                    )

logger = logging.getLogger(__name__)

logger.info("Iniciando programa de optimización con log fechado")

### Manejo de Configuración en YAML ###
logger.info("Configuración cargada desde YAML")
logger.info(f"STUDY_NAME: {STUDY_NAME}")
logger.info(f"DATA_PATH: {DATA_PATH}")
logger.info(f"SEMILLA: {SEMILLA}")
logger.info(f"MES_TRAIN: {MES_TRAIN}")
logger.info(f"MES_VALIDACION: {MES_VALIDACION}")
logger.info(f"MES_TEST: {MES_TEST}")
logger.info(f"GANANCIA_ACIERTO: {GANANCIA_ACIERTO}")
logger.info(f"COSTO_ESTIMULO: {COSTO_ESTIMULO}")


def main():
    """Pipeline principal con optimización usando configuración YAML."""
    logger.info("=== INICIANDO OPTIMIZACIÓN CON CONFIGURACIÓN YAML ===")
  
    # 1. Cargar datos
    df = cargar_datos(DATA_PATH)

 
    # 2. Feature Engineering
    #col_out = ["clase_ternaria"] 
    #atributos = list(set(df.columns).difference(col_out))
    atributos = ["mcuentas_saldo", "mtarjeta_visa_consumo", "cproductos"] 
    cant_lag = 2
    df_fe = feature_engineering_lag(df, atributos, cant_lag)
    logger.info(f"Feature Engineering completado: {df_fe.shape}")
  
    # 3. Convertir clase_ternaria a binario
    df_fe = convertir_clase_ternaria_a_target(df_fe)
   
#    # 4. Ejecutar optimización (función simple)
#    study = optimizar(df_fe, n_trials=HIPERPARAM_BO['N_TRIALS'])
#  
#    # 5. Análisis adicional
#    logger.info("=== ANÁLISIS DE RESULTADOS ===")
#    trials_df = study.trials_dataframe()
#    if len(trials_df) > 0:
#        top_5 = trials_df.nlargest(5, 'value')
#        logger.info("Top 5 mejores trials:")
#        for idx, trial in top_5.iterrows():
#            logger.info(f"  Trial {trial['number']}: {trial['value']:,.0f}")
#  
#    logger.info("=== OPTIMIZACIÓN COMPLETADA ===")
    
    # 6 Test en mes desconocido
    logger.info("=== EVALUACIÓN EN CONJUNTO DE TEST ===")
    # Cargar mejores hiperparámetros
    mejores_params = cargar_mejores_hiperparametros()
  
    # Evaluar en test
    resultados_test = evaluar_en_test(df_fe, mejores_params)
  
    # Guardar resultados de test
    guardar_resultados_test(resultados_test)
  
    # Resumen de evaluación en test
    logger.info("=== RESUMEN DE EVALUACIÓN EN TEST ===")
    logger.info(f"✅ Ganancia en test: {resultados_test['ganancia_test']:,.0f}, MSE: {resultados_test['mse']:.4f}")
    logger.info(f"🎯 Predicciones positivas: {resultados_test['predicciones_positivas']:,} ({resultados_test['porcentaje_positivas']:.2f}%)")


    # 7 Entrenar modelo final
    logger.info("=== ENTRENAMIENTO FINAL ===")
    logger.info("Preparar datos para entrenamiento final")
    X_train, y_train, X_predict, clientes_predict = preparar_datos_entrenamiento_final(df_fe)
  
    # Entrenar modelo final
    logger.info("Entrenar modelo final")
    modelo_final = entrenar_modelo_final(X_train, y_train, mejores_params)
  
    # Generar predicciones finales
    logger.info("Generar predicciones finales")
    resultados = generar_predicciones_finales(modelo_final, X_predict, clientes_predict)
#  
    # Guardar predicciones
    logger.info("Guardar predicciones")
    archivo_salida = guardar_predicciones_finales(resultados)
#  
    # Resumen final
    logger.info("=== RESUMEN FINAL ===")
    logger.info(f"✅ Entrenamiento final completado exitosamente")
    logger.info(f"📊 Mejores hiperparámetros utilizados: {mejores_params}")
    logger.info(f"🎯 Períodos de entrenamiento: {FINAL_TRAIN}")
    logger.info(f"🔮 Período de predicción: {FINAL_PREDIC}")
    logger.info(f"📁 Archivo de salida: {archivo_salida}")
#    logger.info(f"📝 Log detallado: logs/{monbre_log}")
##
##
    logger.info(f">>> Ejecución finalizada. Revisar logs para mas detalles.")

 
   
if __name__ == "__main__":
    main()