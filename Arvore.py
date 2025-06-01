import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Criando dados
data = {
    "tipo_solo": ["arenoso", "argiloso", "siltoso"] * 333 + ["arenoso"],
    "clima": ["seco", "umido", "temperado"] * 333 + ["temperado"],
    "demanda_mercado": ["alta", "media", "baixa"] * 333 + ["alta"],
    "cultura": []  # Será preenchida com base nas regras
}

# regras para determinar a cultura
cultura = []
for solo, clima, demanda in zip(data["tipo_solo"], data["clima"], data["demanda_mercado"]):
    if demanda == "alta":
        if solo == "arenoso":
            cultura.append("milho")
        elif solo == "argiloso":
            cultura.append("soja")
        else:
            cultura.append("trigo")
    elif solo == "arenoso" and clima == "temperado":
        cultura.append("milho")
    elif solo == "argiloso" and clima == "umido":
        cultura.append("soja")
    else:
        cultura.append("trigo")

data["cultura"] = cultura
df = pd.DataFrame(data)

# Convertendo variáveis categóricas em variáveis dummy
df_encoded = pd.get_dummies(df, columns=["tipo_solo", "clima", "demanda_mercado"], drop_first=True)

# variável de saída
label_encoder = LabelEncoder()
df["cultura"] = label_encoder.fit_transform(df["cultura"])

X = df_encoded.drop("cultura", axis=1)  # Atributos de entrada
y = df["cultura"]  # Variável alvo

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train, y_train)

y_pred = decision_tree.predict(X_test)
print(f"Acurácia da Árvore de Decisão: {accuracy_score(y_test, y_pred):.2f}")
print(classification_report(y_test, y_pred, zero_division=0))

plt.figure(figsize=(50, 25))
plot_tree(
    decision_tree,
    feature_names=X.columns,
    class_names=label_encoder.classes_,
    filled=True,
    fontsize=10,
    rounded=True,
    proportion=False
)
plt.title("Árvore de Decisão - Planejamento de Cultivo Agrícola", fontsize=16)
plt.show()