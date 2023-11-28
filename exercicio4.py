import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Criando o dataframe com os dados
data = {
    'Substância': ['Álcool', 'Gasolina', 'Leite', 'Querosene', 'Óleo', 'Vinho'],
    'Concentração (%)': [12.5, 0.1, 4.0, 1.2, 0.5, 15.0],
    'Teor Alcoólico (%)': [50, 0.05, 0.01, 0.02, 0.01, 12.5]
}

df = pd.DataFrame(data)

# Exibindo o dataframe
print(df)

# Selecionando apenas as colunas relevantes para o agrupamento
X = df[['Concentração (%)', 'Teor Alcoólico (%)']]

# Definindo o número de clusters desejado (vamos usar 2 clusters neste exemplo)
num_clusters = 2

# Aplicando o algoritmo K-Means
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(X)

# Adicionando os rótulos dos clusters ao DataFrame
df['Cluster'] = kmeans.labels_

# Exibindo o DataFrame com os rótulos dos clusters
print(df)

# Visualização dos clusters
plt.scatter(X['Concentração (%)'], X['Teor Alcoólico (%)'], c=kmeans.labels_, cmap='viridis', edgecolor='k')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', marker='*', label='Centroids')
plt.xlabel('Concentração (%)')
plt.ylabel('Teor Alcoólico (%)')
plt.title('Agrupamento de Substâncias')
plt.legend()
plt.show()
