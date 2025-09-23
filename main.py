import pandas as pd
import os
import datetime

def main():
    print("Inicio de ejecución del programa")
    
    try:
        # Cargar datos
       df = pd.read_csv('data/competencia_01_crudo.csv')
    except Exception as e:
        print("Error: El archivo no se encuentra.")
        return
    print(df.head())
    print(f"Filas: {df.shape[0]}, Columnas: {df.shape[1]}")
    print("Fin de ejecución del programa")
    
    with open("logs/logs.txt", "a") as f:
        f.write(f"Ejecución finalizada: {datetime.datetime.now()}\n")
    
if __name__ == "__main__":
    main()