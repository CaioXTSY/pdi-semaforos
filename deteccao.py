"""Detecção das luzes e classificação do estado do semáforo."""

import math

import cv2
import numpy as np

ESTADOS = {
    "vermelho": "VERMELHO",
    "amarelo": "AMARELO",
    "verde": "VERDE",
}


def encontrar_contornos(mascara):
    """Encontra os contornos externos de uma máscara binária."""
    contornos, _ = cv2.findContours(
        mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return contornos


def calcular_circularidade(contorno):
    """Calcula 4πA/P²; valores próximos de 1 indicam círculos."""
    area = cv2.contourArea(contorno)
    perimetro = cv2.arcLength(contorno, True)
    return 4 * math.pi * area / perimetro**2 if perimetro else 0


def filtrar_candidatos(contornos, area_minima, circularidade_minima=0.45):
    """Mantém regiões grandes o suficiente e aproximadamente circulares."""
    return [
        contorno
        for contorno in contornos
        if cv2.contourArea(contorno) >= area_minima
        and calcular_circularidade(contorno) >= circularidade_minima
    ]


def classificar_estado(
    mascaras, proporcao_minima=0.002, circularidade_minima=0.45
):
    """Escolhe a cor com melhor proporção de pixels na zona esperada."""
    deteccoes = []

    for cor, mascara in mascaras.items():
        area_regiao = mascara.size / 3
        area_minima = max(8, area_regiao * 0.0005)
        candidatos = filtrar_candidatos(
            encontrar_contornos(mascara), area_minima, circularidade_minima
        )
        if not candidatos:
            continue

        melhor = max(candidatos, key=cv2.contourArea)
        mascara_valida = np.zeros_like(mascara)
        cv2.drawContours(mascara_valida, candidatos, -1, 255, cv2.FILLED)
        pixels = cv2.countNonZero(cv2.bitwise_and(mascara, mascara_valida))
        proporcao = pixels / area_regiao
        if proporcao < proporcao_minima:
            continue

        deteccoes.append(
            {
                "cor": cor,
                "proporcao": proporcao,
                "circularidade": calcular_circularidade(melhor),
                "regiao": cv2.boundingRect(melhor),
            }
        )

    if not deteccoes:
        return {"estado": "DESCONHECIDO", "confianca": 0.0, "regiao": None}

    melhor = max(deteccoes, key=lambda item: item["proporcao"])
    total = sum(item["proporcao"] for item in deteccoes)
    dominancia = melhor["proporcao"] / total
    confianca = (dominancia + melhor["circularidade"]) / 2

    return {
        "estado": ESTADOS[melhor["cor"]],
        "confianca": round(min(confianca, 1.0), 2),
        "regiao": melhor["regiao"],
    }
