"""Detecção das luzes e classificação do estado do semáforo."""

import math

import cv2
import numpy as np

ESTADOS = {
    "vermelho": "VERMELHO",
    "amarelo": "AMARELO",
    "verde": "VERDE",
}

CORES_ESTADO = {
    "VERMELHO": (0, 0, 255),
    "AMARELO": (0, 255, 255),
    "VERDE": (0, 255, 0),
    "DESCONHECIDO": (255, 255, 255),
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


def criar_painel(imagem, titulo, largura=320, altura=210):
    """Prepara uma imagem com proporção preservada e título."""
    if imagem.ndim == 2:
        imagem = cv2.cvtColor(imagem, cv2.COLOR_GRAY2BGR)

    escala = min(largura / imagem.shape[1], (altura - 30) / imagem.shape[0])
    tamanho = (
        max(1, int(imagem.shape[1] * escala)),
        max(1, int(imagem.shape[0] * escala)),
    )
    imagem = cv2.resize(imagem, tamanho, interpolation=cv2.INTER_AREA)
    painel = np.full((altura, largura, 3), 25, dtype=np.uint8)
    x = (largura - tamanho[0]) // 2
    y = 30 + (altura - 30 - tamanho[1]) // 2
    painel[y : y + tamanho[1], x : x + tamanho[0]] = imagem
    cv2.putText(
        painel, titulo, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1
    )
    return painel


def mostrar_resultado(
    frame, recorte, imagem_processada, mascaras, mascaras_limpas, roi, resultado
):
    """Exibe todas as etapas em uma única janela."""
    x, y, largura, altura = roi
    original = frame.copy()
    cv2.rectangle(original, (x, y), (x + largura, y + altura), (255, 255, 255), 2)
    texto = f"Estado: {resultado['estado']} ({resultado['confianca']:.0%})"
    cor_estado = CORES_ESTADO[resultado["estado"]]
    cv2.putText(
        original, texto, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, cor_estado, 2
    )

    detectada = imagem_processada.copy()
    if resultado["regiao"]:
        rx, ry, rw, rh = resultado["regiao"]
        cv2.rectangle(
            detectada, (rx, ry), (rx + rw, ry + rh), cor_estado, 2
        )

    paineis = [
        criar_painel(original, "Quadro original"),
        criar_painel(recorte, "Regiao selecionada"),
        criar_painel(detectada, texto),
    ]
    paineis += [
        criar_painel(mascaras[cor], f"{cor.capitalize()} - segmentada")
        for cor in mascaras
    ]
    paineis += [
        criar_painel(mascaras_limpas[cor], f"{cor.capitalize()} - limpa")
        for cor in mascaras_limpas
    ]

    linhas = [np.hstack(paineis[i : i + 3]) for i in range(0, 9, 3)]
    cv2.imshow("Detector de semaforo", np.vstack(linhas))
