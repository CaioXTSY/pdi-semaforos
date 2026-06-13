"""Funções de recorte e pré-processamento da região do semáforo."""

import cv2
import numpy as np

TAMANHO_PADRAO = (200, 400)

# Limites HSV do OpenCV: H entre 0-179; S e V entre 0-255.
LIMITES_HSV = {
    "vermelho": [
        ((0, 100, 70), (10, 255, 255)),
        ((170, 100, 70), (179, 255, 255)),
    ],
    "amarelo": [((15, 100, 70), (35, 255, 255))],
    "verde": [((40, 70, 70), (90, 255, 255))],
}


def redimensionar(frame, largura=960):
    """Redimensiona o quadro mantendo sua proporção."""
    if frame.shape[1] <= largura:
        return frame

    escala = largura / frame.shape[1]
    return cv2.resize(frame, None, fx=escala, fy=escala)


def selecionar_roi(frame, coordenadas=None):
    """Seleciona uma ROI pelo mouse ou usa (x, y, largura, altura)."""
    if coordenadas is None:
        coordenadas = cv2.selectROI("Selecione o semaforo", frame, False)
        cv2.destroyWindow("Selecione o semaforo")

    x, y, largura, altura = map(int, coordenadas)
    if largura == 0 or altura == 0:
        raise ValueError("Região de interesse inválida.")

    return frame[y : y + altura, x : x + largura], (x, y, largura, altura)


def filtro_gaussiano(imagem):
    """Suaviza ruídos gerais usando uma média ponderada."""
    return cv2.GaussianBlur(imagem, (5, 5), 0)


def filtro_mediana(imagem):
    """Reduz ruídos pontuais preservando melhor as bordas."""
    return cv2.medianBlur(imagem, 5)


def preprocessar(imagem, filtro="mediana"):
    """Padroniza o tamanho e aplica o filtro escolhido."""
    imagem = cv2.resize(imagem, TAMANHO_PADRAO, interpolation=cv2.INTER_AREA)

    if filtro == "gaussiano":
        return filtro_gaussiano(imagem)
    if filtro == "mediana":
        return filtro_mediana(imagem)

    raise ValueError("Filtro deve ser 'gaussiano' ou 'mediana'.")


def converter_hsv(imagem):
    """Converte uma imagem BGR para o espaço de cores HSV."""
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


def criar_kernel(tamanho=5):
    """Cria um elemento estruturante elíptico."""
    if tamanho <= 0:
        raise ValueError("O tamanho do kernel deve ser maior que zero.")
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tamanho, tamanho))


def erodir(mascara, kernel, iteracoes=1):
    return cv2.erode(mascara, kernel, iterations=iteracoes)


def dilatar(mascara, kernel, iteracoes=1):
    return cv2.dilate(mascara, kernel, iterations=iteracoes)


def abertura(mascara, kernel, iteracoes=1):
    return cv2.morphologyEx(
        mascara, cv2.MORPH_OPEN, kernel, iterations=iteracoes
    )


def fechamento(mascara, kernel, iteracoes=1):
    return cv2.morphologyEx(
        mascara, cv2.MORPH_CLOSE, kernel, iterations=iteracoes
    )


def limpar_mascara(mascara, tamanho_kernel=5, iteracoes=1):
    """Remove pontos isolados e preenche pequenas falhas."""
    if iteracoes <= 0:
        raise ValueError("A quantidade de iterações deve ser maior que zero.")

    kernel = criar_kernel(tamanho_kernel)
    mascara = abertura(mascara, kernel, iteracoes)
    return fechamento(mascara, kernel, iteracoes)


def limpar_mascaras(mascaras, tamanho_kernel=5, iteracoes=1):
    """Aplica a limpeza morfológica a todas as máscaras."""
    return {
        cor: limpar_mascara(mascara, tamanho_kernel, iteracoes)
        for cor, mascara in mascaras.items()
    }
