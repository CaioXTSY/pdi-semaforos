"""Leitura de imagens e vídeos e seleção da região do semáforo."""

import argparse
from pathlib import Path

import cv2

from processamento import redimensionar, selecionar_roi


def mostrar(frame, recorte, roi):
    x, y, largura, altura = roi
    original = frame.copy()
    cv2.rectangle(original, (x, y), (x + largura, y + altura), (255, 255, 255), 2)
    cv2.imshow("Quadro original", original)
    cv2.imshow("Regiao do semaforo", recorte)


def processar_imagem(caminho, coordenadas):
    frame = cv2.imread(str(caminho))
    if frame is None:
        return False

    frame = redimensionar(frame)
    recorte, roi = selecionar_roi(frame, coordenadas)
    mostrar(frame, recorte, roi)
    cv2.waitKey(0)
    return True


def processar_video(caminho, coordenadas):
    video = cv2.VideoCapture(str(caminho))
    if not video.isOpened():
        raise ValueError("Não foi possível abrir a entrada.")

    roi = coordenadas
    while True:
        sucesso, frame = video.read()
        if not sucesso:
            break

        frame = redimensionar(frame)
        recorte, roi = selecionar_roi(frame, roi)
        mostrar(frame, recorte, roi)

        if cv2.waitKey(25) & 0xFF in (ord("q"), 27):
            break

    video.release()


def main():
    parser = argparse.ArgumentParser(description="Recorta a região de um semáforo.")
    parser.add_argument("entrada", type=Path)
    parser.add_argument("--roi", nargs=4, type=int, metavar=("X", "Y", "L", "A"))
    args = parser.parse_args()

    if not args.entrada.is_file():
        parser.error("arquivo inexistente ou inválido")

    try:
        if not processar_imagem(args.entrada, args.roi):
            processar_video(args.entrada, args.roi)
    except ValueError as erro:
        parser.error(str(erro))
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
