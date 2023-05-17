import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline 
from sklearn.preprocessing import StandardScaler 

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
#Se lee el archivo CSV y se carga en un DataFrame de pandas:
df = pd.read_csv('coords.csv')
#Se separan las características (X) y la variable objetivo (y) del DataFrame:
X = df.drop('class', axis=1) 
y = df['class']
#Se divide el conjunto de datos en conjuntos de entrenamiento y prueba utilizando train_test_split:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, train_size= 0.5, random_state=1234)
#postura del triangulo "triangle pose"
#postura del warrior pose "warrior pose"
#postura del arbol "tree pose"
df[df['class']=='warrior pose']
#Se define un diccionario de pipelines, donde cada pipeline aplica una serie de transformaciones a los datos antes de ajustar un modelo específico:
pipelines = {
    'lr':make_pipeline(StandardScaler(), LogisticRegression(solver='lbfgs', max_iter=10000)),
    'rc':make_pipeline(StandardScaler(), RidgeClassifier()),
    'rf':make_pipeline(StandardScaler(), RandomForestClassifier()),
    'gb':make_pipeline(StandardScaler(), GradientBoostingClassifier()),
}
#Se entrena cada modelo en el conjunto de entrenamiento y se almacena en un diccionario llamado fit_models:
fit_models = {}
print( X_test)
for algo, pipeline in pipelines.items():
    model = pipeline.fit(X_train, y_train)
    fit_models[algo] = model
#Se realiza una predicción utilizando el modelo de clasificador Ridge (rc) en los datos de prueba:
print(fit_models['rc'].predict(X_test))

from sklearn.metrics import accuracy_score
import pickle 
#Se evalúa la precisión de cada modelo utilizando la métrica de precisión (accuracy_score) en los datos de prueba:
for algo, model in fit_models.items():
    yhat = model.predict(X_test)
    print(algo, accuracy_score(y_test, yhat))
#Se guarda el modelo de clasificación de bosque aleatorio (rf) en un archivo utilizando pickle:
with open('exercises.pkl', 'wb') as f:
    pickle.dump(fit_models['rf'], f)