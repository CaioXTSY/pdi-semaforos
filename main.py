"""Entrada, processamento e detecção do estado do semáforo."""

import argparse
from pathlib import Path

import cv2

from deteccao import classificar_estado, mostrar_resultado
from entrada import redimensionar, selecionar_roi
from processamento import preprocessar
from segmentacao import segmentar


def processar_imagem(
    caminho, coordenadas, filtro, kernel=3, iteracoes=1, ampliacao=4
):
    frame = cv2.imread(str(caminho))
    if frame is None:
        return False

    frame = redimensionar(frame)
    recorte, roi = selecionar_roi(frame, coordenadas)
    imagem_processada = preprocessar(recorte, filtro, ampliacao)
    mascaras, mascaras_limpas = segmentar(imagem_processada, kernel, iteracoes)
    resultado = classificar_estado(mascaras_limpas)
    mostrar_resultado(
        frame,
        recorte,
        imagem_processada,
        mascaras,
        mascaras_limpas,
        roi,
        resultado,
    )
    print(f"Estado do semáforo: {resultado['estado']}")
    cv2.waitKey(0)
    return True


def processar_video(
    caminho, coordenadas, filtro, kernel=3, iteracoes=1, ampliacao=4
):
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
        imagem_processada = preprocessar(recorte, filtro, ampliacao)
        mascaras, mascaras_limpas = segmentar(
            imagem_processada, kernel, iteracoes
        )
        resultado = classificar_estado(mascaras_limpas)
        mostrar_resultado(
            frame,
            recorte,
            imagem_processada,
            mascaras,
            mascaras_limpas,
            roi,
            resultado,
        )

        if cv2.waitKey(25) & 0xFF in (ord("q"), 27):
            break

    video.release()


def main():
    parser = argparse.ArgumentParser(description="Detecta o estado de um semáforo.")
    parser.add_argument("entrada", type=Path)
    parser.add_argument("--roi", nargs=4, type=int, metavar=("X", "Y", "L", "A"))
    parser.add_argument(
        "--filtro",
        choices=("mediana", "gaussiano"),
        default="mediana",
        help="Filtro de redução de ruído (padrão: mediana).",
    )
    parser.add_argument(
        "--kernel",
        type=int,
        default=3,
        help="Tamanho do kernel morfológico (padrão: 3).",
    )
    parser.add_argument(
        "--iteracoes",
        type=int,
        default=1,
        help="Iterações das operações morfológicas (padrão: 1).",
    )
    parser.add_argument(
        "--ampliacao",
        type=float,
        default=4,
        help="Ampliação da ROI para semáforos distantes (padrão: 4).",
    )
    args = parser.parse_args()

    if not args.entrada.is_file():
        parser.error("arquivo inexistente ou inválido")

    try:
        if not processar_imagem(
            args.entrada,
            args.roi,
            args.filtro,
            args.kernel,
            args.iteracoes,
            args.ampliacao,
        ):
            processar_video(
                args.entrada,
                args.roi,
                args.filtro,
                args.kernel,
                args.iteracoes,
                args.ampliacao,
            )
    except ValueError as erro:
        parser.error(str(erro))
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
