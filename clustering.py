import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Dados fictícios
data = {
    'estacao': ['Capinzal-norte', 'Escola-Municipal-Ybacanga', 'Posto-saúde-Bacanga', 'Gapara', 'Estação-AERCA', 'UTE-Interna', 'Santo-Antonio-lopes', 'Areias II', 'Areias', 'Machadinho', 'Caboto', 'Botelho', 'Câmara', 'Cobre', 'Leandrinho', 'Gravatá', 'Lamarao', 'Futurama I', 'Malemba', 'Gamboa', 'Concordia', 'Escola', 'EDCUPE', 'IFPE', 'Ipojuca', 'CPRH', 'Portoalegre/CETE'],
    'PM10': [33, 18, 64, 24, 27, 20, 33, 0, 0, 0, 23, 0, 0, 0, 12, 10, 0, 0, 7, 5, 7, 0, 0, 0, 34, 27, 0],
    'PM2.5': [0, 15, 21, 0, 19, 0, 0, 0, 0, 15, 0, 0, 0, 0, 10, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 11]
}

# Mapeamento de cidades e estados
estados = {
    'Capinzal-norte': 'MA', 'Escola-Municipal-Ybacanga': 'MA', 'Posto-saúde-Bacanga': 'MA', 'Gapara': 'MA', 'Estação-AERCA': 'MA', 'UTE-Interna': 'MA', 'Santo-Antonio-lopes': 'MA',
    'Areias II': 'BA', 'Areias': 'BA', 'Machadinho': 'BA', 'Caboto': 'BA', 'Botelho': 'BA', 'Câmara': 'BA', 'Cobre': 'BA', 'Leandrinho': 'BA', 'Gravatá': 'BA', 'Lamarao': 'BA', 'Futurama I': 'BA', 'Malemba': 'BA', 'Gamboa': 'BA', 'Concordia': 'BA', 'Escola': 'BA',
    'EDCUPE': 'PE', 'IFPE': 'PE', 'Ipojuca': 'PE', 'Portoalegre/CETE': 'RN'
}

cidades = {
    'Capinzal-norte': 'São Luís', 'Escola-Municipal-Ybacanga': 'São Luís', 'Posto-saúde-Bacanga': 'São Luís', 'Gapara': 'São Luís', 'Estação-AERCA': 'Açailândia', 'UTE-Interna': 'São Luís', 'Santo-Antonio-lopes': 'Santo Antônio',
    'Areias II': 'Camaçari', 'Areias': 'Camaçari', 'Machadinho': 'Camaçari', 'Caboto': 'Candeias', 'Botelho': 'Salvador', 'Câmara': 'Dias d’Ávila', 'Cobre': 'Dias d’Ávila', 'Leandrinho': 'Dias d’Ávila', 'Gravatá': 'Camaçari',
    'Lamarao': 'São Sebastião do Passo', 'Futurama I': 'Dias d’Ávila', 'Malemba': 'Camaçari', 'Gamboa': 'Candeias', 'Concordia': 'Dias d’Ávila', 'Escola': 'Dias d’Ávila',
    'EDCUPE': 'Ipojuca', 'IFPE': 'Ipojuca', 'Ipojuca': 'Cabo de Santo Agostinho', 'Portoalegre/CETE': 'Porto Alegre'
}

# Criando DataFrame
df = pd.DataFrame(data)
df['Estado'] = df['estacao'].map(estados)
df['Cidade'] = df['estacao'].map(cidades)

# Normalização dos dados
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[['PM10', 'PM2.5']])

# Aplicação do KMeans
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(df_scaled)

# Tabela final com Cluster incluído
df['Poluição Total'] = df['PM10'] + df['PM2.5']
cidade_menos_poluida = df.loc[df['Poluição Total'].idxmin(), ['estacao', 'Cidade', 'Estado']]
cidade_mais_poluida = df.loc[df['Poluição Total'].idxmax(), ['estacao', 'Cidade', 'Estado']]

# Exibir tabela corrigida
print(df[['estacao', 'Cidade', 'Estado', 'PM10', 'PM2.5', 'Cluster']])

# Exibir as cidades mais e menos poluídas
print(f"\nA cidade com menos poluição é: {cidade_menos_poluida['estacao']} ({cidade_menos_poluida['Cidade']}, {cidade_menos_poluida['Estado']})")
print(f"A cidade com mais poluição é: {cidade_mais_poluida['estacao']} ({cidade_mais_poluida['Cidade']}, {cidade_mais_poluida['Estado']})")

# Gráfico
plt.scatter(df['PM2.5'], df['PM10'], c=df['Cluster'], cmap='viridis')
plt.xlabel('PM2.5')
plt.ylabel('PM10')
plt.title('Clusters de Poluição do Ar')
plt.colorbar(label='Cluster')
plt.show()
