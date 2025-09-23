import pandas as pd
import duckdb
import logging

logger = logging.getLogger("__name__")

def feature_engineering_lag(df: pd.DataFrame, attributos: list[str], cant_lag: int = 1) -> pd.DataFrame:
    """
    Crea variables rezagadas (lags) para las columnas especificadas en el DataFrame.

    Args:
        df (pd.DataFrame): DataFrame de entrada.
        attributos (list[str]): Lista de nombres de columnas para las cuales se crearán los lags.
        cant_lag (int, optional): Número de períodos de rezago. Por defecto es 1.

    Returns:
        pd.DataFrame: DataFrame con las nuevas columnas de lags añadidas.
    """
    logger.info(f"Creando variables rezagadas para las columnas: {attributos} con {cant_lag} períodos de rezago.")
    
    if attributos is None or len(attributos) == 0:
        logger.warning("No se especificaron columnas para crear lags. Retornando el DataFrame original.")
        return df
    
    # Consulta SQL
    sql = "SELECT *"
    
    # Agregar los lags para las columnas que queremos
    for attr in attributos:
        if attr in df.columns:
            for i in range(1, cant_lag + 1):
                sql += f", LAG({attr}, {i}) OVER (ORDER BY foto_mes) AS {attr}_lag_{i}"
        else:
            logger.warning(f"La columna '{attr}' no existe en el DataFrame. Se omitirá la creación de lags para esta columna.")
            
    
    # Completar la consulta SQL        
    sql += " FROM df"
    
    logger.debug(f"Consulta SQL generada: {sql}")
    
    
    # Ejecutar la consulta SQL usando DuckDB
    
    con = duckdb.connect(database=':memory:')
    con.register('df', df)
    df = con.execute(sql).df()
    con.close()
    
    
    print(df.head())
    logger.info(f"Variables rezagadas creadas exitosamente. Dataset resultante tiene {df.shape[0]} filas y {df.shape[1]} columnas.")
    
    return df