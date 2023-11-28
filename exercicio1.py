import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

# Criando o DataFrame com os dados dos vinhos
data = {
    'Teor Alcoólico': [3, 4, 5, 6],
    'Acidez': ['muito', 'pouco', 'médio', 'baixo'],
    'pH': [4.3, 2.8, 4.2, 3.9]
}

wine_df = pd.DataFrame(data)

# Mapeando os valores de Acidez para valores numéricos
acidity_mapping = {
    'muito': 3,
    'médio': 2,
    'pouco': 1,
    'baixo': 0
}
wine_df['Acidez'] = wine_df['Acidez'].map(acidity_mapping)

# Criando a matriz numpy com os dados de Teor Alcoólico, Acidez, pH
wine_features = wine_df[['Teor Alcoólico', 'Acidez', 'pH']].values

print("Matriz numpy dos dados:")
print(wine_features)
