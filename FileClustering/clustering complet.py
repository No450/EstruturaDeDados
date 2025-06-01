import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# tabela completa
pd.set_option("display.max_columns", None)  # colunas
pd.set_option("display.max_rows", None)  # linhas
pd.set_option("display.width", 1000)  # exibição

# Carregar os dados
df = pd.read_csv("projeto_cluster.csv")

# Calcular o total de poluentes
df['Total_poluentes'] = df['PM10'] + df['PM2,5'] + df['O3'] + df['NO2'] + df['SO2'] + df['CO']

# Selecionando as colunas relevantes
colunas_cluster = ['PM10', 'PM2,5', 'O3', 'NO2', 'SO2', 'CO', 'IQAR', 'Total_poluentes']
X = df[colunas_cluster]

# Normalizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicando KMeans com 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_scaled)

# Adicionando os clusters ao DataFrame
df['Cluster'] = kmeans.labels_

# Salvando os dados com clusters em um arquivo CSV
df.to_csv("cluster_resultado_kmeans.csv", index=False)

# Imprimindo a tabela completa
print("\nTabela completa com clusters:")
print(df)

# Plotando os clusters (IQAR vs Total de Poluentes) em 2D
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Total_poluentes', y='IQAR', hue='Cluster', palette='viridis')
plt.xlabel("Total de Poluentes")
plt.ylabel("Índice de Qualidade do Ar (IQAR)")
plt.title("Clusters Identificados pelo KMeans (IQAR vs Total de Poluentes)")
plt.legend(title="Cluster")
plt.show()

# Exibindo estatísticas dos clusters
status_cluster = df.groupby('Cluster').describe()
print("\nEstatísticas dos clusters:")
print(status_cluster)

# Criando um mapa de calor (heatmap) e exibindo a tabela de correlação para cada cluster
for cluster in df['Cluster'].unique():
    print(f"\nAnálise do Cluster {cluster}:")

    # Filtrar os dados do cluster atual
    cluster_data = df[df['Cluster'] == cluster][colunas_cluster]

    # Calcular a matriz de correlação
    correlation_matrix = cluster_data.corr()

    # Exibir a tabela de correlação
    print(f"\nTabela de Correlação - Cluster {cluster}:")
    print(correlation_matrix)

    # Plotar o mapa de calor
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title(f"Mapa de Calor de Correlação - Cluster {cluster}")
    plt.show()
