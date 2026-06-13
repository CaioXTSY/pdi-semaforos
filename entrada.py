"""Etapa 1: preparação da entrada e seleção da região de interesse."""

import argparse
from pathlib import Path

import cv2


def ler_argumentos():
    parser = argparse.ArgumentParser(description="Detecta o estado de um semáforo.")
    parser.add_argument("entrada", type=Path)
    parser.add_argument("--roi", nargs=4, type=int, metavar=("X", "Y", "L", "A"))
    parser.add_argument(
        "--filtro", choices=("mediana", "gaussiano"), default="mediana"
    )
    parser.add_argument("--kernel", type=int, default=3)
    parser.add_argument("--iteracoes", type=int, default=1)
    parser.add_argument("--ampliacao", type=float, default=4)
    argumentos = parser.parse_args()

    if not argumentos.entrada.is_file():
        parser.error("arquivo inexistente ou inválido")

    return argumentos


def ler_quadros(caminho):
    """Produz os quadros e informa se a entrada é uma imagem."""
    imagem = cv2.imread(str(caminho))
    if imagem is not None:
        yield imagem, True
        return

    video = cv2.VideoCapture(str(caminho))
    if not video.isOpened():
        raise ValueError("Não foi possível abrir a entrada.")

    try:
        while True:
            sucesso, frame = video.read()
            if not sucesso:
                break
            yield frame, False
    finally:
        video.release()


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
