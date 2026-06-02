import numpy as np
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════
# HACKATHON — Sistema de Alerta Escolar com ML
# Autora: Viviane Caroline (Nawi)
# Data: maio 2026
#
# Problema: identificar alunos em risco ANTES
# que seja tarde para intervir
#
# Conceitos: vetores, matrizes, PCA, estatística,
# correlação, normalização, visualização
# ═══════════════════════════════════════════════════

# ─── DATASET — 30 ALUNOS ─────────────────────────
# [matemática, português, ciências, história, faltas]
nomes = [
    "Ana","Bruno","Carla","Diego","Elena",
    "Felipe","Gabi","Hugo","Iris","João",
    "Karen","Lucas","Maria","Nelson","Olívia",
    "Paulo","Quezia","Rafael","Sara","Tiago",
    "Ursula","Vitor","Wanda","Xavier","Yara",
    "Zeca","Alice","Beto","Cris","Dani"
]

np.random.seed(42)
dados = np.array([
    [9,8,9,7,2],[5,7,6,8,8],[8,9,7,9,1],
    [4,4,5,4,15],[7,8,8,7,3],[9,6,8,6,2],
    [3,5,4,5,18],[8,7,9,8,1],[6,9,7,9,4],
    [5,4,5,4,12],[7,8,7,8,2],[9,9,8,9,0],
    [4,5,4,4,16],[8,7,8,7,3],[6,8,7,8,5],
    [3,4,3,4,20],[7,7,8,7,4],[9,8,9,8,1],
    [5,6,5,5,10],[8,8,7,8,2],[4,4,4,3,17],
    [9,7,9,7,1],[6,7,6,7,6],[5,5,4,5,11],
    [8,9,8,9,2],[4,3,4,4,19],[7,8,7,8,3],
    [9,9,9,8,0],[6,6,5,6,7],[5,5,6,5,9]
])

materias = ["Matemática","Português","Ciências","História"]

# ─── ESTATÍSTICAS ────────────────────────────────
medias = dados[:, :4].mean(axis=1)
faltas = dados[:, 4]

print("=" * 55)
print("   SISTEMA DE ALERTA ESCOLAR — RELATÓRIO COMPLETO")
print("=" * 55)

# Correlação faltas × notas
corr = np.corrcoef(medias, faltas)[0,1]
print(f"\nCorrelação faltas × média: {corr:.2f}")
print("(quanto mais falta, menor a nota — confirmado)\n")

# ─── CLASSIFICAÇÃO DE RISCO ──────────────────────
print(f"{'Aluno':<10} {'Média':>6} {'Faltas':>7} {'Status':>12}")
print("-" * 40)

em_risco    = []
em_atencao  = []
aprovados   = []

for i, nome in enumerate(nomes):
    m = medias[i]
    f = faltas[i]
    if m < 5 or f > 12:
        status = "🔴 RISCO"
        em_risco.append(nome)
    elif m < 7 or f > 6:
        status = "🟡 ATENÇÃO"
        em_atencao.append(nome)
    else:
        status = "🟢 OK"
        aprovados.append(nome)
    print(f"{nome:<10} {m:>6.1f} {f:>7} {status:>12}")

print(f"\n{'='*55}")
print(f"🔴 Em risco:    {len(em_risco):2} alunos — {em_risco}")
print(f"🟡 Atenção:     {len(em_atencao):2} alunos")
print(f"🟢 OK:          {len(aprovados):2} alunos")
print(f"{'='*55}")

# ─── MÉDIA POR MATÉRIA ───────────────────────────
print("\nMédia da turma por matéria:")
for i, mat in enumerate(materias):
    m = dados[:, i].mean()
    barra = "█" * int(m)
    print(f"  {mat:<12} {m:.1f} {barra}")

# ─── PCA — VISUALIZAÇÃO 2D ───────────────────────
media_col = dados.mean(axis=0)
dados_c   = dados - media_col
cov       = np.cov(dados_c.T)
vals, vecs = np.linalg.eig(cov)
ordem     = np.argsort(vals)[::-1]
comp      = vecs[:, ordem[:2]]
dados_2d  = dados_c @ comp

# ─── GRÁFICOS ────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Sistema de Alerta Escolar — Análise ML', fontsize=14)

# Gráfico 1 — Barras por status
cores_bar = []
for i in range(len(nomes)):
    if medias[i] < 5 or faltas[i] > 12:
        cores_bar.append('#D85A30')
    elif medias[i] < 7 or faltas[i] > 6:
        cores_bar.append('#BA7517')
    else:
        cores_bar.append('#1D9E75')

axes[0].bar(range(len(nomes)), medias, color=cores_bar, alpha=0.85)
axes[0].axhline(7, color='#1D9E75', linestyle='--', alpha=0.6, label='Aprovado (7)')
axes[0].axhline(5, color='#D85A30', linestyle='--', alpha=0.6, label='Risco (5)')
axes[0].set_title('Média por Aluno')
axes[0].set_xlabel('Alunos')
axes[0].set_ylabel('Média')
axes[0].legend(fontsize=8)
axes[0].set_xticks([])

# Gráfico 2 — Scatter faltas × média
cores_sc = ['#D85A30' if m<5 or f>12 else '#BA7517' if m<7 or f>6
            else '#1D9E75' for m,f in zip(medias, faltas)]
axes[1].scatter(faltas, medias, c=cores_sc, s=80, alpha=0.85)
axes[1].set_title(f'Faltas × Média (corr={corr:.2f})')
axes[1].set_xlabel('Faltas')
axes[1].set_ylabel('Média')
axes[1].grid(True, alpha=0.3)

# Gráfico 3 — PCA 2D
cores_pca = ['#D85A30' if m<5 or f>12 else '#BA7517' if m<7 or f>6
             else '#1D9E75' for m,f in zip(medias, faltas)]
axes[2].scatter(dados_2d[:,0], dados_2d[:,1], c=cores_pca, s=80, alpha=0.85)
for i in range(0, len(nomes), 5):
    axes[2].annotate(nomes[i],
                     (dados_2d[i,0], dados_2d[i,1]),
                     fontsize=7, alpha=0.7)
axes[2].set_title('PCA — Perfil dos Alunos em 2D')
axes[2].set_xlabel('Componente 1')
axes[2].set_ylabel('Componente 2')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('alerta_escolar.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n✅ Gráfico salvo como alerta_escolar.png")
print("✅ Projeto pronto para o GitHub!")