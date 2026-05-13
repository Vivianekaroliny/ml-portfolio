import numpy as np
import matplotlib.pyplot as plt

# ─── DATASET REAL ─────────────────────────────────
# 10 alunos · [matemática, português, ciências, história, faltas]
nomes = ["Ana","Bruno","Carla","Diego","Elena",
         "Felipe","Gabi","Hugo","Iris","João"]

dados = np.array([
    [9, 8, 9, 7, 2],   # Ana
    [5, 7, 6, 8, 8],   # Bruno
    [8, 9, 7, 9, 1],   # Carla
    [4, 4, 5, 4, 15],  # Diego — risco
    [7, 8, 8, 7, 3],   # Elena
    [9, 6, 8, 6, 2],   # Felipe
    [3, 5, 4, 5, 18],  # Gabi — risco
    [8, 7, 9, 8, 1],   # Hugo
    [6, 9, 7, 9, 4],   # Iris
    [5, 4, 5, 4, 12],  # João — atenção
])

# ─── MÉDIAS E ALERTAS ─────────────────────────────
medias = dados[:, :4].mean(axis=1)
faltas = dados[:, 4]

print("=== RELATÓRIO DE RISCO ===")
for i, nome in enumerate(nomes):
    if medias[i] < 5 or faltas[i] > 12:
        status = "🔴 RISCO"
    elif medias[i] < 7 or faltas[i] > 6:
        status = "🟡 ATENÇÃO"
    else:
        status = "🟢 OK"
    print(f"{nome:8} | média {medias[i]:.1f} | faltas {faltas[i]:2} | {status}")

# ─── PCA: reduzir para 2D ─────────────────────────
media_col = dados.mean(axis=0)
dados_c   = dados - media_col
cov       = np.cov(dados_c.T)
vals, vecs = np.linalg.eig(cov)
ordem     = np.argsort(vals)[::-1]
comp      = vecs[:, ordem[:2]]
dados_2d  = dados_c @ comp

# ─── GRÁFICO ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
cores_plot = []
for i in range(len(nomes)):
    if medias[i] < 5 or faltas[i] > 12:
        cores_plot.append('red')
    elif medias[i] < 7 or faltas[i] > 6:
        cores_plot.append('orange')
    else:
        cores_plot.append('#1D9E75')

ax.scatter(dados_2d[:, 0], dados_2d[:, 1],
           c=cores_plot, s=120, alpha=0.85, zorder=3)

for i, nome in enumerate(nomes):
    ax.annotate(nome, (dados_2d[i, 0], dados_2d[i, 1]),
                textcoords="offset points",
                xytext=(6, 4), fontsize=9)

ax.set_title('Perfil de Alunos — PCA (verde=OK · laranja=atenção · vermelho=risco)')
ax.set_xlabel('Componente Principal 1')
ax.set_ylabel('Componente Principal 2')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
print("\n✅ Projeto 3 concluído!")