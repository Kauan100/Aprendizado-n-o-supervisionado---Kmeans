import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt  # Importe corrigido para matplotlib

# Dados fornecidos
setores = [1, 2, 3, 4, 5, 6, 7, 8]
produtos_fabricados = [100, 50, 15, 200, 500, 1000, 375, 450]

# Criando o DataFrame
dados = {'Setores': setores, 'Produtos Fabricados': produtos_fabricados}
df = pd.DataFrame(dados)

# Defina o número de clusters desejados
num_clusters = 3  # Por exemplo, pode ser alterado

# Usando o algoritmo KMeans
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(df[['Produtos Fabricados']])

# Adicionando os resultados ao DataFrame
df['Cluster'] = kmeans.labels_

# Plotando os clusters
cores = ['blue', 'green', 'red', 'purple', 'orange', 'yellow', 'brown', 'black']
plt.scatter(df['Setores'], df['Produtos Fabricados'], c=df['Cluster'], cmap=plt.cm.get_cmap('tab10', num_clusters))
plt.xlabel('Setores')
plt.ylabel('Produtos Fabricados')
plt.title('Clusters dos Setores mais Produtivos')
plt.show()
