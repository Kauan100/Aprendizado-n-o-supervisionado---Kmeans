import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Dados fornecidos
data = np.array([
    [70.2, 12.5, 4.7],
    [65.1, 8.2, 3.9],
    [75.5, 15.6, 5.1],
    [80.3, 10.2, 4.5],
    [68.7, 11.8, 4.2],
    [72.9, 14.3, 5.3],
    [78.6, 9.8, 4.8],
    [66.4, 8.9, 4.0],
    [73.1, 13.7, 5.0],
    [69.5, 12.1, 4.3]
])

# Aplicando o algoritmo K-Means com 3 clusters (pode-se ajustar o número de clusters conforme necessário)
kmeans = KMeans(n_clusters=3)
kmeans.fit(data)

# Obtendo os rótulos dos clusters e adicionando-os aos dados
clusters = kmeans.labels_
data_with_clusters = np.column_stack((data, clusters))

# Plotando boxplots para cada cluster
plt.figure(figsize=(8, 6))
plt.boxplot([data_with_clusters[data_with_clusters[:, -1] == i, :-1] for i in range(kmeans.n_clusters)])
plt.xlabel('Features')
plt.ylabel('Valores')
plt.title('Boxplot dos Clusters')
plt.xticks([1, 2, 3], ['Temperatura', 'Vibração', 'Corrente'])
plt.grid(True)
plt.show()
