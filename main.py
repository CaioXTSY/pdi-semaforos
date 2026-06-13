"""Entrada, seleção da região do semáforo e pré-processamento."""

import argparse
from pathlib import Path

import cv2

from processamento import preprocessar, redimensionar, selecionar_roi


def mostrar(frame, recorte, imagem_processada, roi):
    x, y, largura, altura = roi
    original = frame.copy()
    cv2.rectangle(original, (x, y), (x + largura, y + altura), (255, 255, 255), 2)
    cv2.imshow("Quadro original", original)
    cv2.imshow("Regiao original", recorte)
    cv2.imshow("Regiao pre-processada", imagem_processada)


def processar_imagem(caminho, coordenadas, filtro):
    frame = cv2.imread(str(caminho))
    if frame is None:
        return False

    frame = redimensionar(frame)
    recorte, roi = selecionar_roi(frame, coordenadas)
    imagem_processada = preprocessar(recorte, filtro)
    mostrar(frame, recorte, imagem_processada, roi)
    cv2.waitKey(0)
    return True


def processar_video(caminho, coordenadas, filtro):
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
        imagem_processada = preprocessar(recorte, filtro)
        mostrar(frame, recorte, imagem_processada, roi)

        if cv2.waitKey(25) & 0xFF in (ord("q"), 27):
            break

    video.release()


def main():
    parser = argparse.ArgumentParser(description="Pré-processa a região de um semáforo.")
    parser.add_argument("entrada", type=Path)
    parser.add_argument("--roi", nargs=4, type=int, metavar=("X", "Y", "L", "A"))
    parser.add_argument(
        "--filtro",
        choices=("mediana", "gaussiano"),
        default="mediana",
        help="Filtro de redução de ruído (padrão: mediana).",
    )
    args = parser.parse_args()

    if not args.entrada.is_file():
        parser.error("arquivo inexistente ou inválido")

    try:
        if not processar_imagem(args.entrada, args.roi, args.filtro):
            processar_video(args.entrada, args.roi, args.filtro)
    except ValueError as erro:
        parser.error(str(erro))
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
