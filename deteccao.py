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

POSICOES_ESPERADAS = {
    "vermelho": 0.20,
    "amarelo": 0.50,
    "verde": 0.80,
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


def avaliar_candidato(contorno, largura, altura, posicao_esperada):
    """Pontua forma, preenchimento e posição de uma possível luz."""
    area = cv2.contourArea(contorno)
    x, y, largura_regiao, altura_regiao = cv2.boundingRect(contorno)
    centro_x = (x + largura_regiao / 2) / largura
    centro_y = (y + altura_regiao / 2) / altura

    # A luz deve ficar no miolo horizontal da ROI, não na moldura lateral.
    if not 0.15 <= centro_x <= 0.85:
        return None

    proporcao_area = area / (largura * altura)
    if proporcao_area < 0.001:
        return None

    proporcao_forma = min(largura_regiao, altura_regiao) / max(
        largura_regiao, altura_regiao
    )
    if proporcao_forma < 0.45:
        return None

    preenchimento = area / (largura_regiao * altura_regiao)
    circularidade = calcular_circularidade(contorno)
    centralidade = 1 - abs(centro_x - 0.5) / 0.35
    coerencia_vertical = 1 - abs(centro_y - posicao_esperada) / 0.35
    centralidade = max(0.0, min(centralidade, 1.0))
    coerencia_vertical = max(0.0, min(coerencia_vertical, 1.0))

    forma = (
        0.30 * circularidade
        + 0.20 * proporcao_forma
        + 0.20 * preenchimento
        + 0.15 * centralidade
        + 0.15 * coerencia_vertical
    )
    intensidade = min(proporcao_area / 0.01, 1.0)
    confianca = 0.75 * forma + 0.25 * intensidade

    return {
        "confianca": confianca,
        "regiao": (x, y, largura_regiao, altura_regiao),
    }


def classificar_estado(
    mascaras, confianca_minima=0.50
):
    """Escolhe o candidato mais coerente com uma luz de semáforo."""
    deteccoes = []

    for cor, mascara in mascaras.items():
        altura, largura = mascara.shape
        for contorno in encontrar_contornos(mascara):
            candidato = avaliar_candidato(
                contorno, largura, altura, POSICOES_ESPERADAS[cor]
            )
            if candidato is not None:
                candidato["cor"] = cor
                deteccoes.append(candidato)

    if not deteccoes:
        return {"estado": "DESCONHECIDO", "confianca": 0.0, "regiao": None}

    melhor = max(deteccoes, key=lambda item: item["confianca"])
    if melhor["confianca"] < confianca_minima:
        return {"estado": "DESCONHECIDO", "confianca": 0.0, "regiao": None}

    return {
        "estado": ESTADOS[melhor["cor"]],
        "confianca": round(min(melhor["confianca"], 1.0), 2),
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
