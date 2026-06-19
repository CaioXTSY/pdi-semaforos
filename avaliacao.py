"""Métricas simples e salvamento dos resultados."""

import csv
from pathlib import Path


def calcular_acuracia(reais, previstos):
    """Calcula a proporção de classificações corretas."""
    if len(reais) != len(previstos):
        raise ValueError("As listas devem ter o mesmo tamanho.")
    if not reais:
        return 0.0
    return sum(real == previsto for real, previsto in zip(reais, previstos)) / len(
        reais
    )


def salvar_resultados(resultados, caminho):
    """Salva uma sequência de dicionários em CSV."""
    if not resultados:
        return

    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with caminho.open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=resultados[0])
        escritor.writeheader()
        escritor.writerows(resultados)
