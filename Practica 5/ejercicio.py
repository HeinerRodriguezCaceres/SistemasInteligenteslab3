# ==========================================
# REGRESIÓN LOGÍSTICA - CÁNCER (NIVEL 20)
# ==========================================

# Librerías
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ==========================================
# 1. CARGA DE DATOS
# ==========================================

data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

print("Dimensiones del dataset:", X.shape)
print("\nDistribución de clases:")
print(y.value_counts())

# ==========================================
# 2. ANÁLISIS EXPLORATORIO (EDA)
# ==========================================

# Histograma de clases
plt.figure()
y.value_counts().plot(kind='bar')
plt.title("Distribución de Tumores (0 = Maligno, 1 = Benigno)")
plt.xlabel("Clase")
plt.ylabel("Cantidad")
plt.show()

# Histograma de variables importantes
plt.figure()
plt.hist(X['mean radius'])
plt.title("Distribución - Radio Medio")
plt.show()

plt.figure()
plt.hist(X['mean texture'])
plt.title("Distribución - Textura Media")
plt.show()

# ==========================================
# 3. PREPROCESAMIENTO
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==========================================
# 4. MODELO
# ==========================================

modelo = LogisticRegression(max_iter=5000)
modelo.fit(X_train, y_train)

# ==========================================
# 5. EVALUACIÓN
# ==========================================

y_pred = modelo.predict(X_test)

print("\n===== RESULTADOS =====")
print("Exactitud:", accuracy_score(y_test, y_pred))

print("\nMatriz de Confusión:")
print(confusion_matrix(y_test, y_pred))

print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred))

# ==========================================
# 6. MATRIZ DE CONFUSIÓN GRÁFICA
# ==========================================

cm = confusion_matrix(y_test, y_pred)

plt.figure()
plt.imshow(cm)
plt.title("Matriz de Confusión")
plt.colorbar()

plt.xlabel("Predicción")
plt.ylabel("Real")

for i in range(len(cm)):
    for j in range(len(cm[0])):
        plt.text(j, i, cm[i][j], ha='center')

plt.show()

# ==========================================
# 7. IMPORTANCIA DE VARIABLES
# ==========================================

coeficientes = pd.DataFrame(
    modelo.coef_[0],
    index=data.feature_names,
    columns=['Importancia']
)

coeficientes = coeficientes.sort_values(by='Importancia', ascending=False)

print("\nTop 10 variables más importantes:")
print(coeficientes.head(10))

# Gráfica de importancia
plt.figure()
coeficientes.head(10).plot(kind='bar')
plt.title("Top 10 Variables Más Influyentes")
plt.ylabel("Peso")
plt.show()

# ==========================================
# 8. PREDICCIÓN DE EJEMPLO
# ==========================================

ejemplo = X.iloc[0:1]
ejemplo_scaled = scaler.transform(ejemplo)

prediccion = modelo.predict(ejemplo_scaled)

print("\nPredicción de ejemplo:", "Benigno" if prediccion[0]==1 else "Maligno")