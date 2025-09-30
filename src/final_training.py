import pandas as pd
import lightgbm as lgb
import numpy as np
import logging
import os
from datetime import datetime
from .conf import FINAL_TRAIN, FINAL_PREDIC, SEMILLA, LGBM_PARAMS_BASE, HIPERPARAM_BO
import copy
from .best_params import cargar_mejores_hiperparametros
from .gain_function import ganancia_lgb_binary

logger = logging.getLogger(__name__)

def preparar_datos_entrenamiento_final(df: pd.DataFrame) -> tuple:
    """
    Prepara los datos para el entrenamiento final usando todos los períodos de FINAL_TRAIN.
  
    Args:
        df: DataFrame con todos los datos
  
    Returns:
        tuple: (X_train, y_train, X_predict, clientes_predict)
    """
    logger.info(f"Preparando datos para entrenamiento final")
    logger.info(f"Períodos de entrenamiento: {FINAL_TRAIN}")
    logger.info(f"Período de predicción: {FINAL_PREDIC}")
  
    # Datos de entrenamiento: todos los períodos en FINAL_TRAIN
    
    df_train = df[df['foto_mes'].isin(FINAL_TRAIN)].copy()
    df_predict = df[df['foto_mes'] == FINAL_PREDIC].copy()
  
    # Datos de predicción: período FINAL_PREDIC 

    logger.info(f"Registros de entrenamiento: {len(df_train):,}")
    logger.info(f"Registros de predicción: {len(df_predict):,}")
  
    #Corroborar que no esten vacios los df

    # Preparar features y target para entrenamiento
  
    X_train, y_train = df_train.drop(columns=['clase_ternaria']), df_train['clase_ternaria'] 

    # Preparar features para predicción
    X_predict = df_predict.drop(columns=['clase_ternaria']) 
    clientes_predict = df_predict['numero_de_cliente'].values
    features_cols = X_predict.columns.tolist()

    logger.info(f"Features utilizadas: {len(features_cols)}")
    logger.info(f"Distribución del target - 0: {(y_train == 0).sum():,}, 1: {(y_train == 1).sum():,}")
  
    return X_train, y_train, X_predict, clientes_predict

def entrenar_modelo_final(X_train: pd.DataFrame, y_train: pd.Series, mejores_params: dict) -> lgb.Booster:
    """
    Entrena el modelo final con los mejores hiperparámetros.
  
    Args:
        X_train: Features de entrenamiento
        y_train: Target de entrenamiento
        mejores_params: Mejores hiperparámetros de Optuna
  
    Returns:
        lgb.Booster: Modelo entrenado
    """
    logger.info("Iniciando entrenamiento del modelo final")
  
  
    # Crear dataset de LightGBM
    dtrain = lgb.Dataset(X_train, label=y_train)
    
    # Indicamos los mejores hiperparámetros
    params_base = LGBM_PARAMS_BASE
    params_final = params_base.copy()
    params_final.update(mejores_params)
    
    # Normalizamos ciertos parámetros
    param_normalizado = copy.deepcopy(params_final)
    param_normalizado['min_data_in_leaf'] = round(params_final['min_data_in_leaf'] / HIPERPARAM_BO['UNDERSUMPLING'])
    
     # Configurar parámetros del modelo
    params = {
        'objective': 'binary',
        'metric': 'None',  # Usamos nuestra métrica personalizada
        'random_state': SEMILLA[0] if isinstance(SEMILLA, list) else SEMILLA,
        'verbose': -1,
        **param_normalizado  # Agregar los mejores hiperparámetros
    }
  
    logger.info(f"Parámetros del modelo: {params}")
    logger.info(f"Comenzando entrenamiento con {len(X_train):,} registros")
    
    # Entrenar modelo con lgb.train()
    modelo = lgb.train(
        params,
        dtrain)
    
    logger.info("Entrenamiento completado")

    return modelo

def generar_predicciones_finales(modelo: lgb.Booster, X_predict: pd.DataFrame, clientes_predict: np.ndarray, umbral: float = 0.025) -> pd.DataFrame:
    """
    Genera las predicciones finales para el período objetivo.
  
    Args:
        modelo: Modelo entrenado
        X_predict: Features para predicción
        clientes_predict: IDs de clientes
        umbral: Umbral para clasificación binaria
  
    Returns:
        pd.DataFrame: DataFrame con numero_de_cliente y predict
    """
    logger.info("Generando predicciones finales")
  
    # Generar probabilidades con el modelo entrenado
    # Predecir en test
    y_pred = modelo.predict(X_predict)
    
    # Lo que en realidad quiero es poder calcular la ganancia para diferentes umbrales. COMPLETAR!
    
    # Convertir a predicciones binarias con el umbral establecido
    y_pred_binary = (y_pred > 0.025).astype(int)
  
    # Crear DataFrame de 'resultados' con nombres de atributos que pide kaggle
    
    resultados = pd.DataFrame({
        'numero_cliente': clientes_predict,
        'predict': y_pred_binary
    })
  
    # Estadísticas de predicciones
    total_predicciones = len(resultados)
    predicciones_positivas = (resultados['predict'] == 1).sum()
    porcentaje_positivas = (predicciones_positivas / total_predicciones) * 100
  
    logger.info(f"Predicciones generadas:")
    logger.info(f"  Total clientes: {total_predicciones:,}")
    logger.info(f"  Predicciones positivas: {predicciones_positivas:,} ({porcentaje_positivas:.2f}%)")
    logger.info(f"  Predicciones negativas: {total_predicciones - predicciones_positivas:,}")
    logger.info(f"  Umbral utilizado: {umbral}")
  
    return resultados
