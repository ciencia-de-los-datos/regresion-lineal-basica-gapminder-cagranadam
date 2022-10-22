df = pd.read_csv("gm_2008_region.csv",
    sep= ",",
    thousands=None
    decimal= ".",
    )

    # Asigne la columna "life" a `y` y la columna "fertility" a `X`
y = df['life'].array
X = df['fertility'].array

    # Imprima las dimensiones de `y`
print(y.shape)

    # Imprima las dimensiones de `X`
print(X.shape)

    # Transforme `y` a un array de numpy usando reshape
y_reshaped = y.reshape(-1 , 1)

    # Trasforme `X` a un array de numpy usando reshape
X_reshaped = X.reshape(-1, 1)

    # Imprima las nuevas dimensiones de `y`
print(y_reshaped.shape)

    # Imprima las nuevas dimensiones de `X`
print(X_reshaped.shape)