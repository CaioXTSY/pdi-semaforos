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


def filtrar_candidatos(contornos, area_minima=100, circularidade_minima=0.6):
    """Mantém regiões grandes o suficiente e aproximadamente circulares."""
    return [
        contorno
        for contorno in contornos
        if cv2.contourArea(contorno) >= area_minima
        and calcular_circularidade(contorno) >= circularidade_minima
    ]


def classificar_estado(
    mascaras, area_minima=100, circularidade_minima=0.6
):
    """Escolhe a cor válida com maior quantidade de pixels ativos."""
    deteccoes = []

    for cor, mascara in mascaras.items():
        candidatos = filtrar_candidatos(
            encontrar_contornos(mascara), area_minima, circularidade_minima
        )
        if not candidatos:
            continue

        melhor = max(candidatos, key=cv2.contourArea)
        mascara_valida = np.zeros_like(mascara)
        cv2.drawContours(mascara_valida, candidatos, -1, 255, cv2.FILLED)
        deteccoes.append(
            {
                "cor": cor,
                "pixels": cv2.countNonZero(
                    cv2.bitwise_and(mascara, mascara_valida)
                ),
                "circularidade": calcular_circularidade(melhor),
                "regiao": cv2.boundingRect(melhor),
            }
        )

    if not deteccoes:
        return {"estado": "DESCONHECIDO", "confianca": 0.0, "regiao": None}

    melhor = max(deteccoes, key=lambda item: item["pixels"])
    total_pixels = sum(item["pixels"] for item in deteccoes)
    dominancia = melhor["pixels"] / total_pixels
    confianca = (dominancia + melhor["circularidade"]) / 2

    return {
        "estado": ESTADOS[melhor["cor"]],
        "confianca": round(min(confianca, 1.0), 2),
        "regiao": melhor["regiao"],
    }
