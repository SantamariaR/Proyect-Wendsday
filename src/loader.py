import pandas as pd
import logging


logger = logging.getLogger("loader")

def cargar_datos(path:str) -> pd.DataFrame | None:
    logger.info(f"Cargando datos desde {path}")
    try:
        df = pd.read_csv(path)
        logger.info(f"Datos de dataset, Filas: {df.shape[0]}, Columnas: {df.shape[1]}")
        return df
    except Exception as e:
        logger.error(f"Error al cargar los datos: {e}")
        raise 