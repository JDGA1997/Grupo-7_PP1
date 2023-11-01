import pandas as pd
import chardet
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# Función para detectar la codificación del archivo
def detect_encoding(file_path):
    rawdata = open(file_path, "rb").read()
    result = chardet.detect(rawdata)
    return result['encoding']

# Ruta al archivo CSV
file_path = '../ventas_electronica.csv'

# Detecta la codificación del archivo
encoding = detect_encoding(file_path)

# Carga el archivo CSV con la codificación detectada
data = pd.read_csv(file_path, encoding=encoding)


#REGRESION LINEAL MODELO CREADO CON VALOR DE VENTA Y COSTO UNITARIO

# Carga de datos desde el archivo CSV con una codificación diferente
data = pd.read_csv('../ventas_electronica.csv', encoding='latin1')

# Elimina los caracteres no numéricos de la columna 'Sales Amount'
data['Sales Amount'] = data['Sales Amount'].str.replace('$', '').str.replace(',', '')

# Convierte la columna a valores numéricos
data['Sales Amount'] = data['Sales Amount'].astype(float)

X = data[['Sales Amount']]
y = data['Unit Cost']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

model = LinearRegression()


model.fit(X_train, y_train)


# Hacer predicciones
y_pred = model.predict(X_test)

# Calcular el R^2 Score
r2 = r2_score(y_test, y_pred)
print(f'R^2 Score (Modelo 1): {r2:.2f}')


#REGRESION LINEAL MODELO CREADO CON VALOR DE VENTA Y COSTO DE TRANSPORTE

# Preparar los datos
X2 = data[['Freight']]
y2 = data['Sales Amount']

# Dividir los datos en conjuntos de entrenamiento y prueba
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

# Instanciar el modelo
model2 = LinearRegression()

# Entrenar el modelo
model2.fit(X2_train, y2_train)

# Hacer predicciones
y2_pred = model2.predict(X2_test)

# Calcular el R^2 Score del  modelo
r2_2 = r2_score(y2_test, y2_pred)
print(f'R^2 Score (Modelo 2): {r2_2:.2f}')


#SE CREA UN MODELO DE REGRESION LASSO CON EL MODELO 2 QUE ES EL SELECCIONADO Y SE LE REALIZA LA HIPERPARAMETRIZACION

from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV

# Modelo Lasso
lasso_model = Lasso(alpha=1.0)  # Puedes ajustar el valor de alpha según tus necesidades
lasso_model.fit(X2_train, y2_train)
y2_lasso_pred = lasso_model.predict(X2_test)
r2_lasso = r2_score(y2_test, y2_lasso_pred)
print(f'R^2 Score (Modelo Lasso): {r2_lasso:.2f}')

# Búsqueda de hiperparámetros para el modelo Lasso
param_grid = {'alpha': [0.1, 0.01, 0.001, 0.0001]}
lasso_model = Lasso()
lasso_grid_search = GridSearchCV(lasso_model, param_grid, scoring='r2', cv=5)
lasso_grid_search.fit(X2_train, y2_train)

# Imprime el mejor valor de alpha para el modelo Lasso
best_alpha = lasso_grid_search.best_params_['alpha']
print(f'El mejor valor de alpha para el modelo Lasso es: {best_alpha}')