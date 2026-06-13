"""Funções de recorte e pré-processamento da região do semáforo."""

import cv2

TAMANHO_PADRAO = (200, 400)


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
