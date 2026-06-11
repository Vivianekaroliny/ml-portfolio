import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# ═══════════════════════════════════════════════════
# Projeto 5 — Comparador de Algoritmos ML
# Autora: Viviane Caroline (Nawi)
# Data: junho 2026
#
# Compara 6 algoritmos ML no mesmo dataset
# Usa cross-validation para resultado mais robusto
# Gera relatório completo e visualização
# ═══════════════════════════════════════════════════

# ─── DATASET — 30 ALUNOS ─────────────────────────
X = np.array([
    [8.5,2],[6.0,8],[9.0,1],[4.0,15],[7.5,3],
    [9.0,2],[3.5,18],[8.5,1],[7.0,4],[5.0,12],
    [8.0,2],[9.5,0],[4.5,16],[8.0,3],[7.0,5],
    [3.0,20],[7.5,4],[9.0,1],[5.5,10],[8.5,2],
    [4.0,17],[9.0,1],[6.5,6],[5.0,11],[8.5,2],
    [4.0,19],[7.5,3],[9.5,0],[6.0,7],[5.5,9]
])
y = np.array([
    1,0,1,0,1,1,0,1,1,0,
    1,1,0,1,1,0,1,1,0,1,
    0,1,1,0,1,0,1,1,1,0
])

# ─── NORMALIZAR ──────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ─── 6 ALGORITMOS ────────────────────────────────
modelos = {
    'Regressão Logística': LogisticRegression(random_state=42),
    'Árvore de Decisão':   DecisionTreeClassifier(max_depth=3, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
    'KNN (K=5)':           KNeighborsClassifier(n_neighbors=5),
    'SVM Linear':          SVC(kernel='linear', random_state=42),
    'SVM RBF':             SVC(kernel='rbf', random_state=42)
}

# ─── CROSS-VALIDATION *(KROS-va-li-DÉI-chon)* ────
# Divide os dados em 5 partes — testa em cada uma
# Resultado mais confiável que treino/teste único
print("=" * 55)
print("   COMPARAÇÃO DE ALGORITMOS — CROSS-VALIDATION")
print("=" * 55)
print(f"\n{'Algoritmo':<25} {'Média':>8} {'Desvio':>8} {'Min':>8} {'Max':>8}")
print("-" * 55)

resultados = {}
for nome, modelo in modelos.items():
    scores = cross_val_score(modelo, X_scaled, y, cv=5)
    resultados[nome] = scores
    print(f"{nome:<25} {scores.mean():>7.1%} "
          f"{scores.std():>7.1%} "
          f"{scores.min():>7.1%} "
          f"{scores.max():>7.1%}")

# ─── MELHOR MODELO ───────────────────────────────
melhor = max(resultados, key=lambda x: resultados[x].mean())
print(f"\n🏆 Melhor modelo: {melhor}")
print(f"   Média CV: {resultados[melhor].mean():.1%}")

# ─── GRÁFICO ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
nomes  = list(resultados.keys())
medias = [resultados[n].mean() for n in nomes]
erros  = [resultados[n].std() for n in nomes]
cores  = ['#7F77DD','#1D9E75','#BA7517',
          '#D85A30','#085041','#3C3489']

bars = ax.barh(nomes, medias, xerr=erros,
               color=cores, alpha=0.85,
               capsize=4, error_kw={'linewidth':1.5})
ax.set_xlim(0, 1.15)
for bar, med in zip(bars, medias):
    ax.text(med+0.02, bar.get_y()+bar.get_height()/2,
            f'{med:.0%}', va='center', fontsize=11,
            fontweight='bold')

ax.set_title('Projeto 5 — Comparação de 6 Algoritmos ML\n(Cross-Validation 5-fold)',
             fontsize=13, pad=12)
ax.set_xlabel('Accuracy média (± desvio padrão)')
ax.axvline(0.8, color='gray', linestyle='--',
           alpha=0.5, label='80% referência')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2, axis='x')
plt.tight_layout()
plt.savefig('projeto05_comparador.png', dpi=150,
            bbox_inches='tight')
plt.show()
print("\n✅ Projeto 5 salvo como projeto05_comparador.png")