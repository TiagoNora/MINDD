import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Carregando o conjunto de dados
df = pd.read_csv("train.csv")

# Visualizando as primeiras linhas do conjunto de dados
print("Primeiras 5 linhas do conjunto de dados:")
print(df.head())

# Exibindo os nomes das colunas
print("\nNomes das colunas:")
print(df.columns)
print(df.dtypes)

# Verificando o tamanho do conjunto de dados (número de linhas e colunas)
print("\nTamanho do conjunto de dados (número de linhas e colunas):")
print(df.shape)

# Verificando o tamanho total do conjunto de dados (número de elementos)
print("\nTamanho total do conjunto de dados (número de elementos):")
print(df.size)

# Exibindo estatísticas descritivas das colunas numéricas
print("\nEstatísticas descritivas das colunas numéricas:")
print(df.describe())

# Exibindo os valores máximos de cada coluna
print("\nValores máximos de cada coluna:")
print(df.max())

# Exibindo os valores minimos de cada coluna
print("\nValores minimos de cada coluna:")
print(df.min())
# Temos de escrever que o status sendo um valor binário nao faz sentido aqui porque é uma variável categórica

missing_data = df.isnull().sum()
print("\nValores ausentes por coluna:")
print(missing_data)

# Remover linhas duplicadas, se houver
df.drop_duplicates(inplace=True)
print("\nNúmero de linhas após remoção de duplicatas:", df.shape[0])

# Exibindo valores NA por coluna
na_values = df.isna().sum()
print("\nValores NA por coluna:")
print(na_values)

#Nao ha NA

fig = plt.figure(figsize = (11,6))
plt.title("Distribution of Target Variable")
df["Loan Status"].value_counts().plot(kind = "pie", autopct = "%1.1f%%", cmap = 'Pastel2')