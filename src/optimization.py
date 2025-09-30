import optuna
import copy
import lightgbm as lgb
import pandas as pd
import numpy as np
import sklearn
import logging
import json
import os
from datetime import datetime
from src.conf import *
from src.gain_function import calcular_ganancia, ganancia_lgb_binary

logger = logging.getLogger(__name__)

def objetivo_ganancia(trial, df) -> float:
    """
    Parameters:
    trial: trial de optuna
    df: dataframe con datos
  
    Description:
    Función objetivo que maximiza ganancia en mes de validación.
    Utiliza configuración YAML para períodos y semilla.
    Define parametros para el modelo LightGBM
    Preparar dataset para entrenamiento y validación
    Entrena modelo con función de ganancia personalizada
    Predecir y calcular ganancia
    Guardar cada iteración en JSON
  
    Returns:
    float: ganancia total
    """
    logger.info(f"Iniciando trial {trial.number} - DataFrame shape: {df.shape}")
    
#    # VERIFICAR EXISTENCIA DE TARGET
#    target_col = 'clase_ternaria'  # o la que uses
#    logger.info(f"Target column '{target_col}' existe: {target_col in df.columns}")
#    
#    if target_col in df.columns:
#        dist = df[target_col].value_counts()
#        logger.info(f"Distribución de {target_col}: {dist.to_dict()}")
#        logger.info(f"Total de filas con target: {len(df)}")
#    else:
#        logger.error(f"Target column '{target_col}' no encontrada")
#        return 0.0
#    
#    # VERIFICAR FILTRADO POR FECHAS
#    if 'foto_mes' in df.columns:
#        meses_unicos = df['foto_mes'].unique()
#        logger.info(f"Meses únicos en datos: {sorted(meses_unicos)}")
#        logger.info(f"MES_TRAIN: {MES_TRAIN}, MES_VALID: {MES_VALIDACION}")
#        
#        # Verificar cuántos datos hay para train
#        train_data = df[df['foto_mes'] == MES_TRAIN]
#        logger.info(f"Filas para MES_TRAIN ({MES_TRAIN}): {len(train_data)}")
#        
#        if len(train_data) == 0:
#            logger.error(f"❌ NO HAY DATOS para MES_TRAIN = {MES_TRAIN}")
#            return 0.0
#    else:
#        logger.warning("No hay columna 'foto_mes' para filtrar por fecha")

    
    # Log de dimensiones iniciales
    logger.info(f"Iniciando trial {trial.number} - DataFrame shape: {df.shape}")

    # ===============================
    # 1. Cargar parámetros base
    # ===============================
    params = LGBM_PARAMS_BASE.copy()
    hs = HIPERPARAM_BO["hs"]

    # 2. Reemplazar hiperparámetros con Optuna
    for hparam, cfg in hs.items():
        tipo = cfg["type"]
        lower, upper = cfg["lower"], cfg["upper"]

        if tipo == "integer":
            params[hparam] = trial.suggest_int(hparam, lower, upper)
        elif tipo == "float":
            if "log" in cfg and cfg["log"]:
                params[hparam] = trial.suggest_float(hparam, lower, upper, log=True)
            else:
                params[hparam] = trial.suggest_float(hparam, lower, upper)

    # ===============================
    # 3. Dataset con undersampling
    # ===============================
    
    if isinstance(MES_TRAIN,list):
        periodos_entrenamiento = MES_TRAIN + MES_VALIDACION
    else:
        periodos_entrenamiento = [MES_TRAIN, MES_VALIDACION]
        
    logger.info(f"Períodos de entrenamiento: {periodos_entrenamiento}")
        
    df_train = df[df["foto_mes"].isin(periodos_entrenamiento)]  # meses a usar
    

    # Separar clases
    df_pos = df_train[df_train["clase_ternaria"] == 1]
    df_neg = df_train[df_train["clase_ternaria"] == 0]

    frac = HIPERPARAM_BO["UNDERSUMPLING"]  
    df_neg_sample = df_neg.sample(frac=frac, random_state=314159)

    df_sub = pd.concat([df_pos, df_neg_sample])
    df_sub = df_sub.sample(frac=1, random_state=314159)  # mezclar

    X, y = df_sub.drop("clase_ternaria", axis=1), df_sub["clase_ternaria"]
    dtrain = lgb.Dataset(X, label=y)
    
    logger.info(f"Dataset de entrenamiento preparado: {X.shape}, Positivos: {y.sum()}, Negativos: {len(y)-y.sum()} (undersampling {frac})")

    # ===============================
    # 4. K-fold cross validation
    # ===============================
    folds = HIPERPARAM_BO["VAL_FOLDS_BO"]

    resultados = lgb.cv(
        params,
        dtrain,
        num_boost_round=params.get("num_iterations", 5000),
        nfold=folds,
        stratified=True,
        shuffle=True,
        feval=ganancia_lgb_binary,
        seed=SEMILLA
        )
    print("Claves disponibles en resultados:", resultados.keys())

    # ===============================
    # 5. Resultado promedio
    # ===============================
    ganancia_promedio = max(resultados["valid ganancia-mean"])

    guardar_iteracion(trial, ganancia_promedio)
    
    logger.debug(f"Trial {trial.number}: Ganancia = {ganancia_promedio:,.0f}")
    return ganancia_promedio



def guardar_iteracion(trial, ganancia, archivo_base=None):
    """
    Guarda cada iteración de la optimización en un único archivo JSON.
  
    Args:
        trial: Trial de Optuna
        ganancia: Valor de ganancia obtenido
        archivo_base: Nombre base del archivo (si es None, usa el de config.yaml)
    """
    if archivo_base is None:
        archivo_base = STUDY_NAME
  
    # Nombre del archivo único para todas las iteraciones
    archivo = f"resultados/{archivo_base}_iteraciones.json"
  
    # Datos de esta iteración
    iteracion_data = {
        'trial_number': trial.number,
        'params': trial.params,
        'value': float(ganancia),
        'datetime': datetime.now().isoformat(),
        'state': 'COMPLETE',  # Si llegamos aquí, el trial se completó exitosamente
        'configuracion': {
            'semilla': SEMILLA,
            'mes_train': MES_TRAIN,
            'mes_validacion': MES_VALIDACION
        }
    }
  
    # Cargar datos existentes si el archivo ya existe
    if os.path.exists(archivo):
        with open(archivo, 'r') as f:
            try:
                datos_existentes = json.load(f)
                if not isinstance(datos_existentes, list):
                    datos_existentes = []
            except json.JSONDecodeError:
                datos_existentes = []
    else:
        datos_existentes = []
  
    # Agregar nueva iteración
    datos_existentes.append(iteracion_data)
  
    # Guardar todas las iteraciones en el archivo
    with open(archivo, 'w') as f:
        json.dump(datos_existentes, f, indent=2)
  
    logger.info(f"Iteración {trial.number} guardada en {archivo}")
    logger.info(f"Ganancia: {ganancia:,.0f}" + "---" + "Parámetros: {params}")


def optimizar(df, n_trials=100) -> optuna.Study:
    """
    Args:
        df: DataFrame con datos
        n_trials: Número de trials a ejecutar
        study_name: Nombre del estudio (si es None, usa el de config.yaml)
  
    Description:
       Ejecuta optimización bayesiana de hiperparámetros usando configuración YAML.
       Guarda cada iteración en un archivo JSON separado. 
       Pasos:
        1. Crear estudio de Optuna
        2. Ejecutar optimización
        3. Retornar estudio

    Returns:
        optuna.Study: Estudio de Optuna con resultados
    """

    study_name = STUDY_NAME
    

    logger.info(f"Iniciando optimización con {n_trials} trials")
    logger.info(f"Configuración: TRAIN={MES_TRAIN}, VALID={MES_VALIDACION}, SEMILLA={SEMILLA}")
  
    # Crear estudio
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",  # queremos maximizar ganancia
        sampler=optuna.samplers.TPESampler(seed=SEMILLA)  # bayesiana
    )

    # Ejecutar optimización
    study.optimize(lambda trial: objetivo_ganancia(trial, df), n_trials=n_trials)

  
    # Resultados
    logger.info(f"Mejor ganancia: {study.best_value:,.0f}")
    logger.info(f"Mejores parámetros: {study.best_params}")
  
  
    return study

def evaluar_en_test(df, mejores_params) -> dict:
    """
    Evalúa el modelo con los mejores hiperparámetros en el conjunto de test.
    Solo calcula la ganancia, sin usar sklearn.
  
    Args:
        df: DataFrame con todos los datos
        mejores_params: Mejores hiperparámetros encontrados por Optuna
  
    Returns:
        dict: Resultados de la evaluación en test (ganancia + estadísticas básicas)
    """
    logger.info("=== EVALUACIÓN EN CONJUNTO DE TEST ===")
    logger.info(f"Período de test: {MES_TEST}")
  
    # Preparar datos de entrenamiento (TRAIN + VALIDACION)
    if isinstance(MES_TRAIN, list):
        periodos_entrenamiento = MES_TRAIN + MES_VALIDACION
    else:
        periodos_entrenamiento = [MES_TRAIN, MES_VALIDACION]
  
    df_train_completo = df[df['foto_mes'].isin(periodos_entrenamiento)]
    df_test = df[df['foto_mes'] == MES_TEST]
  
    # Entrenar modelo con mejores parámetros
    # ... Implementar entrenamiento y test con la logica de entrenamiento FINAL para mayor detalle
    # recordar realizar todos los df necesarios y utilizar lgb.train()
    # Preparamos los datos para el modelo final.
    X,y = df_train_completo.drop("clase_ternaria", axis=1),df_train_completo["clase_ternaria"]
    dtrain = lgb.Dataset(X,label=y)
    
    # Armamos los datos de test
    X_test = df_test.drop("clase_ternaria", axis=1)
    y_test = df_test["clase_ternaria"]
    
    # Indicamos los mejores hiperparámetros
    params_base = LGBM_PARAMS_BASE
    params_final = params_base.copy()
    params_final.update(mejores_params)
    
    # Normalizamos ciertos parámetros
    param_normalizado = copy.deepcopy(params_final)
    param_normalizado['min_data_in_leaf'] = round(params_final['min_data_in_leaf'] / HIPERPARAM_BO['UNDERSUMPLING'])
    
    
    modelo_test = lgb.train(
        param_normalizado,
        dtrain)
    
    # Importancia de variables
    importancia = modelo_test.feature_importance(importance_type='gain')
    nombres = modelo_test.feature_name()
    df_importancia = pd.DataFrame({'feature': nombres, 'importance': importancia})
    df_importancia = df_importancia.sort_values(by='importance', ascending=False)
    df_importancia.to_csv(f"resultados/importancia_{STUDY_NAME}.csv")
    
    # Predecir en test
    y_pred = modelo_test.predict(X_test)
    
    # Lo que en realidad quiero es poder calcular la ganancia para diferentes umbrales. COMPLETAR!
    y_pred_binary = (y_pred > 0.025).astype(int)
  
    # Calcular solo la ganancia
    ganancia_test = calcular_ganancia(y_test, y_pred_binary)
  
    # Estadísticas básicas
    total_predicciones = len(y_pred_binary)
    predicciones_positivas = np.sum(y_pred_binary == 1)
    porcentaje_positivas = (predicciones_positivas / total_predicciones) * 100
  
    resultados = {
        'ganancia_test': float(ganancia_test),
        'total_predicciones': int(total_predicciones),
        'predicciones_positivas': int(predicciones_positivas),
        'porcentaje_positivas': float(porcentaje_positivas)
    }
  
    return resultados

#def guardar_resultados_test(resultados_test, archivo_base=None):
#    """
#    Guarda los resultados de la evaluación en test en un archivo JSON.
#    """
#    # Guarda en resultados/{STUDY_NAME}_test_results.json
#    # ... Implementar utilizando la misma logica que cuando guardamos una iteracion de la Bayesiana
#
    