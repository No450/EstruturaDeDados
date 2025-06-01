import networkx as nx
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import numpy as np

grafo = nx.DiGraph()

grafo.add_edges_from([
    ("192.168.0.1", "192.168.0.2"),
    ("192.168.0.2", "192.168.0.3"),
    ("192.168.0.3", "192.168.0.1"),  # Ciclo malicioso 1
    ("192.168.0.2", "192.168.0.4"),
    ("192.168.0.4", "192.168.0.5"),
    ("192.168.0.5", "192.168.0.6"),
    ("192.168.0.6", "192.168.0.4"),  # Ciclo malicioso 2
    ("192.168.0.7", "192.168.0.8"),
    ("192.168.0.8", "192.168.0.9"),
    ("192.168.0.9", "192.168.0.7"),  # Ciclo malicioso 3
    ("192.168.0.10", "192.168.0.11"),
    ("192.168.0.11", "192.168.0.12"),
    ("192.168.0.12", "192.168.0.10"), # Ciclo malicioso 4
    ("192.168.0.13", "192.168.0.14"),
])

def detectar_ciclos(grafo):
    print("\nDetectando ciclos na rede...")
    ciclos = list(nx.simple_cycles(grafo))
    if ciclos:
        print("Ciclos maliciosos detectados:")
        for i, ciclo in enumerate(ciclos, start=1):
            print(f"Ciclo {i}: {' -> '.join(ciclo)}")
    else:
        print("Nenhum ciclo malicioso detectado.")
    return ciclos

# Função para sinalizar IPs envolvidos em ciclos suspeitos
def sinalizar_ips(ciclos):
    ips_suspeitos = set()
    for ciclo in ciclos:
        ips_suspeitos.update(ciclo)
    print("\nIPs suspeitos envolvidos em ciclos maliciosos:")
    for ip in ips_suspeitos:
        print(ip)

def visualizar_grafo(grafo, ciclos):
    pos = nx.spring_layout(grafo)
    plt.figure(figsize=(12, 10))

    # Desenhar todos os nós e arestas
    nx.draw(grafo, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=800, font_size=10)

    # Destacar os ciclos
    if ciclos:
        for ciclo in ciclos:
            edges = [(ciclo[i], ciclo[(i+1) % len(ciclo)]) for i in range(len(ciclo))]
            nx.draw_networkx_edges(grafo, pos, edgelist=edges, edge_color='red', width=2)

    plt.title("Grafo de Comunicação com Ciclos Maliciosos Destacados")
    plt.show()

# detectando ciclos no grafo
ciclos_detectados = detectar_ciclos(grafo)

# visualizar o grafo com os ciclos destacados
visualizar_grafo(grafo, ciclos_detectados)

# sinalizar os IPs suspeitos
sinalizar_ips(ciclos_detectados)

# aplicando Machine Learning para detecção de anomalias com base nos ciclos
print("\nAplicando Machine Learning para detectar comportamentos anômalos com base em ciclos...")

# criar uma lista de participaçao (0 -> não participa, 1 -> participa)
dados = []
for node in grafo.nodes():
    participa_ciclo = 0
    for ciclo in ciclos_detectados:
        if node in ciclo:
            participa_ciclo = 1
            break
    dados.append([participa_ciclo])

modelo = IsolationForest(contamination=0.1, random_state=42)
anomalias = modelo.fit_predict(dados)

# exibir resultados de anomalias
dados_anomalia = {list(grafo.nodes())[i]: "Normal" if anomalias[i] == -1 else "Anômalo" for i in range(len(grafo.nodes()))}
print("\nResultados de Análise de Anomalias:")
for ip, status in dados_anomalia.items():
    print(f"{ip}: {status}")

# visualizar nós anomalos no grafo
# cor verde para IPs "Normais" e vermelho para "Anômalos"
cores = ["red" if dados_anomalia[node] == "Anômalo" else "green" for node in grafo.nodes()]

# visualizar o grafo
plt.figure(figsize=(12, 10))
nx.draw(grafo, pos=nx.spring_layout(grafo), with_labels=True, node_color=cores, edge_color='gray', node_size=800, font_size=10)
plt.title("Detecção de Anomalias no Grafo de Comunicação com Base em Ciclos")
plt.show()