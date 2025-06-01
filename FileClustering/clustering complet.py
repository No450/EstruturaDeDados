import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Configurando tabela completa
pd.set_option("display.max_columns", None)  # Mostra todas as colunas
pd.set_option("display.max_rows", None)  # Mostra todas as linhas
pd.set_option("display.width", 1000)  # Ajusta a largura da exibição

# Dados fornecidos
data = {
    'ID': [0, 2, 12, 13, 14, 0, 18, 3, 0, 9, 4, 11, 0, 1, 10, 5, 6, 0, 8, 0, 6, 0, 0, 15, 16, 17, 0, 6],
    'quantidade projeto': [0, 2, 12, 13, 14, 0, 18, 3, 0, 9, 4, 11, 0, 1, 10, 5, 6, 0, 8, 0, 6, 0, 0, 15, 16, 17, 0, 6],
    'Nome do projeto': ['Solatio', 'Fortescue', 'Enegix', 'EDP', 'Qair', 'Casa dos Ventos', 'Enterprize Energy', 'Qair', 'White Martins', 'Unigel', 'Porto Central', 'Fortescue', 'Porto Norte Fluminense', 'UFRJ', 'Angra 1 e 2', 'Porto de Santos', 'UNIFEI', 'Shell/Raízen/Hytron/Toyota', 'Yara/Raízen', 'Atlas Agro', 'PTI', 'Copel', 'HUB Begreen - Tio Hugo', 'HUB Begreen - Usina Escola', 'Enerfim', 'White Martins', 'Neoenergia', 'Norte Energia'],
    'Local': ['Parnaíba-PI', 'Porto de Pecém-CE', 'Porto de Pecém-CE', 'Porto de Pecém-CE', 'Porto de Pecém-CE', 'Porto de Pecém-CE', 'Touros-CE', 'Suape-PE', 'Porto de Suape', 'Camaçari-BA', 'Espírito Santos', 'Porto do Açu-RJ', 'UNLOCK', 'Rio de Janeiro', 'Angra-RJ', 'São Paulo', 'Itajubá-MG', 'USP-SP', 'Cubatão-SP', 'Uberaba-MG', 'Itumbiara-GO', 'Curitibas- PR', 'Distrito Industrial', 'Passo Fundo', 'Porto do Rio Grande', 'Porto do Rio Grande', 'Porto do Rio Grande', 'Belo Monte- PA'],
    'Finalidade': ['H2V', 'H2V', 'H2V', 'H2V', 'H2V', 'H2V e NH3', 'H2V', 'H2V', 'H2V', 'NH3V', 'H2V', 'H2V', 'H2V', 'Onibus', 'Purificação de H2', 'H2V', 'H2V', 'Onibus', 'Reforma a vapor BioCH4 e NH3V', 'NH3V', 'H2V', 'Transporte', 'NH3V', 'NH3V', 'NH3V', 'H2V', 'H2V', 'NH3V'],
    'Capacidade(T/ano)': [500000.00, 305505.00, 600000.00, 2000.00, 488000.00, 365000.00, 0, 488000.00, 0, 600000.00, 0, 250000.00, 0, 0, 300, 0, 46.5, 390, 240, 500000.00, 10, 0, 4000.00, 2000.00, 0, 0, 0, 120000.00],
    'Valor(R$ milhões)': [50000.00, 6000.00, 5400.00, 42, 6900.00, 4000.00, 0, 3900.00, 0, 1500.00, 0, 0, 0, 0, 0, 0, 0, 50, 0, 43000.00, 45, 12, 65, 45, 0, 0, 0, 1034.00]
}

# Criando o DataFrame
df = pd.DataFrame(data)

# Selecionando as colunas relevantes para o clustering
colunas_cluster = ['Capacidade(T/ano)', 'Valor(R$ milhões)']
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

# Plotando os clusters (Capacidade vs Valor) em 2D
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Capacidade(T/ano)', y='Valor(R$ milhões)', hue='Cluster', palette='viridis')
plt.xlabel("Capacidade (T/ano)")
plt.ylabel("Valor (R$ milhões)")
plt.title("Clusters Identificados pelo KMeans (Capacidade vs Valor)")
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
