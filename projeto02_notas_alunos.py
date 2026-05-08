import numpy as np
import matplotlib.pyplot as plt

# Alunos como linhas, matérias como colunas
# [matemática, português, ciências, história]
notas = np.array([
    [8.5, 7.0, 9.0, 6.5],  # Ana
    [6.0, 8.5, 7.5, 9.0],  # Bruno
    [9.5, 6.0, 8.0, 7.0],  # Carla
    [4.0, 5.5, 4.5, 5.0],  # Diego — em risco
    [7.0, 9.0, 6.5, 8.5],  # Elena
])

alunos   = ["Ana", "Bruno", "Carla", "Diego", "Elena"]
materias = ["Matemática","Português","Ciências","História"]

# Média por aluno
medias = notas.mean(axis=1)
print("=== RELATÓRIO DE DESEMPENHO ===")
for i, nome in enumerate(alunos):
    situacao = "🔴 RISCO" if medias[i] < 6 else "🟡 ATENÇÃO" if medias[i] < 7 else "🟢 OK"
    print(f"{nome}: média {medias[i]:.1f} — {situacao}")

# Melhor aluno
melhor = alunos[np.argmax(medias)]
print(f"\nMelhor desempenho: {melhor}")

# Média por matéria
print("\n=== MÉDIA POR MATÉRIA ===")
for i, mat in enumerate(materias):
    print(f"{mat}: {notas[:,i].mean():.1f}")

# Gráfico
fig, ax = plt.subplots(figsize=(8,4))
cores = ['green' if m >= 7 else 'orange' if m >= 6 else 'red' for m in medias]
ax.bar(alunos, medias, color=cores, alpha=0.8)
ax.axhline(y=7, color='green', linestyle='--', alpha=0.5, label='Aprovado')
ax.axhline(y=6, color='orange', linestyle='--', alpha=0.5, label='Recuperação')
ax.set_title('Desempenho por Aluno')
ax.set_ylabel('Média')
ax.legend()
plt.tight_layout()
plt.show()