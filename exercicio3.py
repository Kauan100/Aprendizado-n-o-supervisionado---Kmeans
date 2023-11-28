import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Dados
dia_semana = np.array([1, 2, 3, 4, 5, 6, 7])
num_clientes = np.array([15, 20, 30, 100, 350, 500, 700])

# Criando um DataFrame com os dados
dados = pd.DataFrame({'Dia_Semana': dia_semana, 'Clientes': num_clientes})

# Visualizando os dados
print(dados)

# Normalizando os dados
dados_normalizados = dados.copy()
dados_normalizados['Clientes'] = (dados['Clientes'] - dados['Clientes'].mean()) / dados['Clientes'].std()

# Aplicando o algoritmo KMeans para agrupar os dias de maior movimento em 3 clusters
kmeans = KMeans(n_clusters=3)
kmeans.fit(dados_normalizados[['Clientes']])

# Adicionando as informações dos clusters ao DataFrame
dados['Cluster'] = kmeans.labels_

# Visualizando os resultados
print(dados)

# Plotando os clusters
plt.scatter(dados['Dia_Semana'], dados['Clientes'], c=dados['Cluster'], cmap='viridis')
plt.xlabel('Dia da Semana')
plt.ylabel('Número de Clientes')
plt.title('Agrupamento dos Dias de Maior Movimento')
plt.show()
