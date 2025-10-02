import pandas as pd
import os
import logging
from datetime import datetime
from .conf import STUDY_NAME

logger = logging.getLogger(__name__)

def guardar_predicciones_finales(resultados_df: pd.DataFrame, nombre_archivo=None) -> str:
    """
    Guarda las predicciones finales en un archivo CSV en la carpeta predict.
  
    Args:
        resultados_df: DataFrame con numero_de_cliente y predict
        nombre_archivo: Nombre del archivo (si es None, usa STUDY_NAME)
  
    Returns:
        str: Ruta del archivo guardado
    """
    # Crear carpeta predict si no existe
    os.makedirs("predict", exist_ok=True)
  
    # Definir nombre del archivo
    if nombre_archivo is None:
        nombre_archivo = STUDY_NAME
  
    # Agregar timestamp para evitar sobrescribir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ruta_archivo = f"predict/{nombre_archivo}_{timestamp}.csv"
  
    # Validar formato del DataFrame

     # Validar que numero_cliente sea numérico
    if not pd.api.types.is_numeric_dtype(resultados_df['numero_cliente']):
        print("❌ ERROR: numero_cliente debe ser numérico")
        raise ValueError("numero_cliente debe ser numérico")
        # Convertir a numérico si es necesario
        resultados_df['numero_cliente'] = pd.to_numeric(resultados_df['numero_cliente'], errors='coerce')
    else:
        print("✅ numero_cliente es numérico")
   
    
  
    # Validar tipos de datos
    valores_unicos = resultados_df['predict'].unique()
    
    # Verificar que solo contenga 0 y 1
    valores_permitidos = {0, 1}
    valores_encontrados = set(valores_unicos)
    
    if valores_encontrados.issubset(valores_permitidos):
        print("✅ predict contiene solo 0 y 1")
    else:
        print("❌ ERROR: predict contiene valores diferentes de 0 y 1")
        valores_invalidos = valores_encontrados - valores_permitidos
        print(f"Valores inválidos encontrados: {valores_invalidos}")
        raise ValueError("predict contiene valores diferentes de 0 y 1")

  
    # Guardar archivo
    resultados_df.to_csv(ruta_archivo, index=False)
  
    logger.info(f"Predicciones guardadas en: {ruta_archivo}")
    logger.info(f"Formato del archivo:")
    logger.info(f"  Columnas: {list(resultados_df.columns)}")
    logger.info(f"  Registros: {len(resultados_df):,}")
    logger.info(f"  Primeras filas:")
    logger.info(f"{resultados_df.head()}")
  
    return ruta_archivo