"""Etapa 1: preparação da entrada e seleção da região de interesse."""

import cv2


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
