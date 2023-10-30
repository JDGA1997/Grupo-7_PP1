
import pandas as pd
import chardet
import pandas as pd
import matplotlib.pyplot as plt


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

# Tamaño del conjunto de datos
print("Tamaño del conjunto de datos:")
print(data.shape)

# Estadísticas descriptivas
print("\nEstadísticas descriptivas:")
print(data.describe())


# Información general del conjunto de datos
print("Información general del conjunto de datos:")
print(data.info())

# Primeras filas del conjunto de datos
print("\nPrimeras filas del conjunto de datos:")
print(data.head())

# Resumen estadístico
print("\nResumen estadístico del conjunto de datos:")
print(data.describe())

# Valores únicos en cada columna
print("\nValores únicos en cada columna:")
for column in data.columns:
    unique_values = data[column].nunique()
    print(f"{column}: {unique_values} valores únicos")

# Comprobar valores faltantes
print("\nValores faltantes en el conjunto de datos:")
missing_values = data.isnull().sum()
print(missing_values)

# Visualización de los primeros registros del conjunto de datos
print("Primeras filas del conjunto de datos:")
print(data.head())

# Histogramas de las variables
data.hist(bins=20, figsize=(12, 8))
plt.show()


# Carga de datos desde el archivo CSV con una codificación diferente
data = pd.read_csv('../ventas_electronica.csv', encoding='latin1')

# Limpieza y formateo de las columnas 'Sales Amount' y 'Freight'
data['Sales Amount'] = data['Sales Amount'].str.replace('$', '').str.replace(',', '').astype(float)
data['Freight'] = data['Freight'].astype(float)



# Cálculo de la rentabilidad general
total_sales_amount = data['Sales Amount'].sum()
total_freight = data['Freight'].sum()
profit = total_sales_amount - total_freight
transport_cost_efficiency = (total_freight / total_sales_amount) * 100

# Cálculo del costo unitario promedio
average_unit_cost = data['Unit Cost'].sum() / total_sales_amount

# Imprimir resultados
print(f'Ingresos totales: {total_sales_amount:.2f}')
print(f'Gastos totales en transporte: {total_freight:.2f}')
print(f'Rentabilidad general: {profit:.2f}')
print(f'Eficiencia de los costos de transporte (%): {transport_cost_efficiency:.2f}%')
print(f'Costo Unitario Promedio: {average_unit_cost:.2f}')

# Crear un gráfico de barras
labels = ['Ingresos Totales', 'Gastos Totales en Transporte', 'Rentabilidad General', 'Eficiencia de Costos de Transporte', 'Costo Unitario Promedio']
values = [total_sales_amount, total_freight, profit, transport_cost_efficiency, average_unit_cost]

plt.figure(figsize=(10, 6))
plt.bar(labels, values)
plt.ylabel('Valor en dólares')
plt.title('Análisis Financiero')
plt.xticks(rotation=45)
plt.show()

# Crear un gráfico de tarta
plt.figure(figsize=(8, 8))
plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=140)
plt.axis('equal')
plt.title('Distribución de Datos Financieros')
plt.show()

#Grafico de dispersion entre el costo unitario y el precio de venta

x = data['Sales Amount']
y = data['Unit Cost']

# Crear el gráfico de dispersión
plt.scatter(x, y)

# Etiquetas de los ejes
plt.xlabel('Sales Amount')
plt.ylabel('Unit Cost')

# Título del gráfico
plt.title('Gráfico de Dispersión de Precio de Venta vs. Costo Unitario')

# Mostrar el gráfico
plt.show()


#VENTAS POR PAIS

# Crear un gráfico de barras
labels = data['Country'].unique()
values = data.groupby('Country')['Sales Amount'].sum()

plt.figure(figsize=(10, 6))
plt.bar(labels, values)
plt.ylabel('Valor en dólares')
plt.title('Ventas por país')
plt.xticks(rotation=45)
plt.show()

import pandas as pd
import matplotlib.pyplot as plt



# Calcula las ventas totales por país
total_sales_by_country = data.groupby('Country')['Sales Amount'].sum()

# Encuentra el país con las ventas más bajas y más altas
lowest_sales_country = total_sales_by_country.idxmin()
highest_sales_country = total_sales_by_country.idxmax()

# Filtra los datos para obtener las ventas de esos dos países
lowest_sales_data = data[data['Country'] == lowest_sales_country]
highest_sales_data = data[data['Country'] == highest_sales_country]

# Crea el gráfico de burbujas
plt.figure(figsize=(10, 6))

# Burbuja para las ventas más bajas
plt.scatter(lowest_sales_data['Sales Amount'], lowest_sales_data['SalesTax'],
            s=lowest_sales_data['Sales Amount'] / 100, label=f'{lowest_sales_country} (Menor Ventas)')

# Burbuja para las ventas más altas
plt.scatter(highest_sales_data['Sales Amount'], highest_sales_data['SalesTax'],
            s=highest_sales_data['Sales Amount'] / 100, label=f'{highest_sales_country} (Mayor Ventas)')

plt.xlabel('Ventas')
plt.ylabel('Impuestos de Ventas')
plt.title('Diagrama de Burbujas - Ventas más bajas y más altas por país')
plt.legend()

plt.show()

