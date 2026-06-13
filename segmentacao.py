"""Etapa 3: segmentação HSV e operações morfológicas."""

import cv2
import numpy as np

# Limites HSV do OpenCV: H entre 0-179; S e V entre 0-255.
LIMITES_HSV = {
    "vermelho": [
        ((0, 80, 60), (12, 255, 255)),
        ((168, 80, 60), (179, 255, 255)),
    ],
    "amarelo": [((15, 80, 70), (38, 255, 255))],
    "verde": [((38, 60, 50), (90, 255, 255))],
}


def converter_hsv(imagem):
    return cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)


def criar_mascaras(imagem_hsv, limites=None):
    """Cria máscaras binárias para vermelho, amarelo e verde."""
    limites = limites or LIMITES_HSV
    mascaras = {}

    for cor, faixas in limites.items():
        mascara = np.zeros(imagem_hsv.shape[:2], dtype=np.uint8)
        for minimo, maximo in faixas:
            faixa = cv2.inRange(
                imagem_hsv,
                np.array(minimo, dtype=np.uint8),
                np.array(maximo, dtype=np.uint8),
            )
            mascara = cv2.bitwise_or(mascara, faixa)
        mascaras[cor] = mascara

    return mascaras


def aplicar_regioes_esperadas(mascaras):
    """Mantém cada cor em uma zona vertical esperada com sobreposição."""
    altura = next(iter(mascaras.values())).shape[0]
    zonas = {
        "vermelho": (0.00, 0.55),
        "amarelo": (0.20, 0.80),
        "verde": (0.45, 1.00),
    }
    resultado = {}

    for cor, mascara in mascaras.items():
        filtrada = np.zeros_like(mascara)
        inicio, fim = zonas[cor]
        filtrada[int(altura * inicio) : int(altura * fim)] = mascara[
            int(altura * inicio) : int(altura * fim)
        ]
        resultado[cor] = filtrada

    return resultado


def limpar_mascara(mascara, tamanho_kernel=3, iteracoes=1):
    """Aplica fechamento e abertura com kernel elíptico."""
    if tamanho_kernel <= 0 or iteracoes <= 0:
        raise ValueError("Kernel e iterações devem ser maiores que zero.")

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (tamanho_kernel, tamanho_kernel)
    )
    mascara = cv2.morphologyEx(
        mascara, cv2.MORPH_CLOSE, kernel, iterations=iteracoes
    )
    return cv2.morphologyEx(
        mascara, cv2.MORPH_OPEN, kernel, iterations=iteracoes
    )


def segmentar(imagem, tamanho_kernel=3, iteracoes=1):
    """Executa HSV, zonas esperadas e limpeza das três máscaras."""
    mascaras = criar_mascaras(converter_hsv(imagem))
    mascaras = aplicar_regioes_esperadas(mascaras)
    mascaras_limpas = {
        cor: limpar_mascara(mascara, tamanho_kernel, iteracoes)
        for cor, mascara in mascaras.items()
    }
    return mascaras, mascaras_limpas
