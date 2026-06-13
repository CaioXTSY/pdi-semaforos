"""Estabilização temporal e métricas simples da classificação."""

import csv
from collections import Counter
from pathlib import Path


def estabilizar_resultado(historico):
    """Escolhe o estado mais frequente dos últimos quadros."""
    if not historico:
        return {"estado": "DESCONHECIDO", "confianca": 0.0, "regiao": None}

    estado = Counter(item["estado"] for item in historico).most_common(1)[0][0]
    resultados = [item for item in historico if item["estado"] == estado]
    resultado = resultados[-1].copy()
    resultado["confianca"] = round(
        sum(item["confianca"] for item in resultados) / len(resultados), 2
    )
    return resultado


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
