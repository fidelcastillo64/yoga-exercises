# yoga exercises
Logistic Regression (Regresión Logística):
La Regresión Logística es un modelo de clasificación que se utiliza para predecir la probabilidad de que una instancia pertenezca a una clase específica. En este caso, se utiliza el algoritmo LogisticRegression de scikit-learn, que implementa la regresión logística multinomial. La función de activación utilizada es la función logística, y el solver (solucionador) utilizado es 'lbfgs', que es un optimizador basado en el método de BFGS. La regresión logística se aplica después de la etapa de estandarización de características.

Ridge Classifier:
El Ridge Classifier es un modelo de clasificación lineal que utiliza la regresión de Ridge para ajustar los coeficientes del modelo. Es similar a la regresión logística, pero utiliza la regularización de Ridge para evitar el sobreajuste. En este caso, se utiliza el algoritmo RidgeClassifier de scikit-learn. También se aplica después de la etapa de estandarización de características.

Random Forest Classifier (Clasificador de Bosque Aleatorio):
El Random Forest Classifier es un modelo de clasificación basado en ensambles que combina múltiples árboles de decisión. Cada árbol se entrena con una muestra aleatoria de las características y utiliza un subconjunto aleatorio de características en cada división. Al combinar las predicciones de múltiples árboles, se reduce el riesgo de sobreajuste y se mejora la precisión general del modelo. En este caso, se utiliza el algoritmo RandomForestClassifier de scikit-learn.

Gradient Boosting Classifier (Clasificador de Reforzamiento de Gradiente):
El Gradient Boosting Classifier es otro modelo basado en ensambles que combina múltiples modelos de clasificación débiles (en este caso, árboles de decisión) en un modelo más fuerte. A diferencia de los bosques aleatorios, el reforzamiento de gradiente construye los árboles secuencialmente, donde cada nuevo árbol se entrena para corregir los errores cometidos por los árboles anteriores. Esto permite mejorar el rendimiento del modelo a medida que se agregan más árboles. En este caso, se utiliza el algoritmo GradientBoostingClassifier de scikit-learn.

Cada uno de estos modelos se encapsula en un pipeline que también incluye la etapa de estandarización de características utilizando StandardScaler. La estandarización de características es importante para garantizar que todas las características tengan la misma escala y no se vean afectadas por diferentes rangos o distribuciones.

Cada pipeline se entrena utilizando el conjunto de datos de entrenamiento y se guarda en el diccionario fit_models para su uso posterior.

En cada uno de los pipelines, se utiliza StandardScaler para estandarizar las características antes de ajustar el modelo correspondiente. La estandarización se realiza para asegurar que todas las características tengan una escala similar y evitar que características con valores más grandes dominen el proceso de aprendizaje.

Cada modelo se entrena utilizando el pipeline correspondiente y se almacena en el diccionario fit_models. Luego, se utiliza el modelo de Bosque Aleatorio (RandomForestClassifier) para hacer una predicción en los datos de prueba y se guarda en un archivo llamado "exercises.pkl" utilizando pickle.

En resumen, los modelos utilizados en los pipelines son Regresión Logística, Clasificador de Ridge, Clasificador de Bosque Aleatorio y Clasificador de Reforzamiento de Gradiente. Cada modelo tiene sus propias características y se utilizan para realizar la clasificación en el conjunto de datos.
