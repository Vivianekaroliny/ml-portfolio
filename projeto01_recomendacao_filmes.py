import numpy as np

# ─── DATASET DE FILMES ───────────────────────────
# Cada filme = vetor [ação, romance, comédia, terror]
filmes = {
    "Vingadores":    np.array([10, 1, 3, 1]),
    "Diário de Uma Paixão": np.array([1, 10, 2, 0]),
    "Se Beber Não Case":    np.array([2, 3, 10, 1]),
    "Arquivo X":            np.array([5, 2, 1, 9]),
    "Homem de Ferro":       np.array([9, 2, 5, 1]),
    "Amor de Primavera":    np.array([1, 9, 3, 0]),
    "Homem-Aranha": np.array([9, 4, 3, 1]),
    "X-Men": np.array([9, 2, 1, 2]),
}

# ─── FUNÇÃO DE SIMILARIDADE ──────────────────────
def similaridade_cosseno(v1, v2):
    return np.dot(v1, v2) / (
        np.linalg.norm(v1) * np.linalg.norm(v2)
    )

# ─── SISTEMA DE RECOMENDAÇÃO ─────────────────────
def recomendar(filme_escolhido, filmes, top=2):
    vetor_base = filmes[filme_escolhido]
    scores = {}
    for nome, vetor in filmes.items():
        if nome != filme_escolhido:
            scores[nome] = similaridade_cosseno(
                vetor_base, vetor
            )
    # Ordenar do mais parecido para o menos
    recomendados = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )
    print(f"\nVocê assistiu: {filme_escolhido}")
    print("Recomendamos:")
    for nome, score in recomendados[:top]:
        print(f"  → {nome} (similaridade: {score:.2f})")

# ─── TESTAR ──────────────────────────────────────
recomendar("Vingadores", filmes)
recomendar("Diário de Uma Paixão", filmes)