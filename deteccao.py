"""Localização das luzes e classificação do estado do semáforo."""

import math

import cv2

CONFIANCA_MINIMA = 0.50
ESTADOS = {
    "vermelho": "VERMELHO",
    "amarelo": "AMARELO",
    "verde": "VERDE",
}
POSICOES_VERTICAIS = {
    "vermelho": 0.20,
    "amarelo": 0.50,
    "verde": 0.80,
}


def encontrar_contornos(mascara):
    contornos, _ = cv2.findContours(
        mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return contornos


def calcular_circularidade(contorno):
    """Retorna 4πA/P²; valores próximos de 1 indicam círculos."""
    area = cv2.contourArea(contorno)
    perimetro = cv2.arcLength(contorno, True)
    return 4 * math.pi * area / perimetro**2 if perimetro else 0


def limitar(valor):
    return max(0.0, min(valor, 1.0))


def avaliar_candidato(contorno, tamanho_mascara, posicao_esperada):
    """Pontua um contorno por tamanho, formato e posição."""
    altura, largura = tamanho_mascara
    x, y, w, h = cv2.boundingRect(contorno)
    area = cv2.contourArea(contorno)
    centro_x = (x + w / 2) / largura
    centro_y = (y + h / 2) / altura

    proporcao_area = area / (largura * altura)
    proporcao_forma = min(w, h) / max(w, h)
    if not 0.15 <= centro_x <= 0.85:
        return None
    if proporcao_area < 0.001 or proporcao_forma < 0.45:
        return None

    criterios = (
        0.30 * calcular_circularidade(contorno)
        + 0.20 * proporcao_forma
        + 0.20 * (area / (w * h))
        + 0.15 * limitar(1 - abs(centro_x - 0.5) / 0.35)
        + 0.15 * limitar(1 - abs(centro_y - posicao_esperada) / 0.35)
    )
    confianca = 0.75 * criterios + 0.25 * limitar(proporcao_area / 0.01)
    return {"confianca": confianca, "regiao": (x, y, w, h)}


def classificar_estado(mascaras, confianca_minima=CONFIANCA_MINIMA):
    """Retorna a cor do candidato mais parecido com uma luz ativa."""
    candidatos = []

    for cor, mascara in mascaras.items():
        for contorno in encontrar_contornos(mascara):
            candidato = avaliar_candidato(
                contorno, mascara.shape, POSICOES_VERTICAIS[cor]
            )
            if candidato:
                candidatos.append({"cor": cor, **candidato})

    if not candidatos:
        return resultado_desconhecido()

    melhor = max(candidatos, key=lambda item: item["confianca"])
    if melhor["confianca"] < confianca_minima:
        return resultado_desconhecido()

    return {
        "estado": ESTADOS[melhor["cor"]],
        "confianca": round(melhor["confianca"], 2),
        "regiao": melhor["regiao"],
    }


def resultado_desconhecido():
    return {"estado": "DESCONHECIDO", "confianca": 0.0, "regiao": None}
