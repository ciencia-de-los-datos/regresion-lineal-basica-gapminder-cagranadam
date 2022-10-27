
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

    # Lea el archivo `gm_2008_region.csv` y asignelo al DataFrame `df`
df = pd.read_csv("gm_2008_region.csv")

    # Asigne a la variable los valores de la columna `fertility`
X_fertility = df['fertility'].values.reshape(139, 1)

    # Asigne a la variable los valores de la columna `life`
y_life = df['life'].values.reshape(139, 1)

    # Divida los datos de entrenamiento y prueba. La semilla del generador de números
    # aleatorios es 53. El tamaño de la muestra de entrenamiento es del 80%
(X_train, X_test, y_train, y_test,) = train_test_split(
        X_fertility,
        y_life,
        test_size=0.2,
        random_state=53,
)

    # Cree una instancia del modelo de regresión lineal
linearRegression = LinearRegression()

    # Entrene el clasificador usando X_train y y_train
linearRegression.fit(X_train, y_train)

    # Pronostique y_test usando X_test
y_pred = linearRegression.predict(X_test)

    # Compute and print R^2 and RMSE
print("R^2: {:6.4f}".format(linearRegression.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error: {:6.4f}".format(rmse))