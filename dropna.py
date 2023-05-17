#codigo no utilizado
import pandas as pd

# Leer el archivo CSV
df = pd.read_csv('coords.csv')

# Eliminar las filas con valores NaN
df = df.dropna()

# Recorrer las filas restantes
for index, row in df.iterrows():
    # Aquí puedes hacer lo que necesites con cada fila
    # Por ejemplo, imprimir el valor de la primera columna:
    print(row[0])
print("listo")
print