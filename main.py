"""Entrada, pré-processamento e segmentação das cores do semáforo."""

import argparse
from pathlib import Path

import cv2

from deteccao import classificar_estado
from processamento import (
    converter_hsv,
    criar_mascaras,
    limpar_mascaras,
    preprocessar,
    redimensionar,
    selecionar_roi,
)

CORES_ESTADO = {
    "VERMELHO": (0, 0, 255),
    "AMARELO": (0, 255, 255),
    "VERDE": (0, 255, 0),
    "DESCONHECIDO": (255, 255, 255),
}


def mostrar(
    frame, recorte, imagem_processada, mascaras, mascaras_limpas, roi, resultado
):
    x, y, largura, altura = roi
    original = frame.copy()
    cv2.rectangle(original, (x, y), (x + largura, y + altura), (255, 255, 255), 2)
    texto = f"Estado: {resultado['estado']} ({resultado['confianca']:.0%})"
    cor = CORES_ESTADO[resultado["estado"]]
    cv2.putText(original, texto, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, cor, 2)

    detectada = imagem_processada.copy()
    if resultado["regiao"]:
        rx, ry, rw, rh = resultado["regiao"]
        cv2.rectangle(detectada, (rx, ry), (rx + rw, ry + rh), cor, 2)

    cv2.imshow("Quadro original", original)
    cv2.imshow("Regiao original", recorte)
    cv2.imshow("Deteccao", detectada)
    for cor in mascaras:
        cv2.imshow(f"Mascara {cor}", mascaras[cor])
        cv2.imshow(f"Mascara {cor} limpa", mascaras_limpas[cor])


def segmentar(imagem, kernel, iteracoes):
    mascaras = criar_mascaras(converter_hsv(imagem))
    return mascaras, limpar_mascaras(mascaras, kernel, iteracoes)


def processar_imagem(caminho, coordenadas, filtro, kernel=5, iteracoes=1):
    frame = cv2.imread(str(caminho))
    if frame is None:
        return False

    frame = redimensionar(frame)
    recorte, roi = selecionar_roi(frame, coordenadas)
    imagem_processada = preprocessar(recorte, filtro)
    mascaras, mascaras_limpas = segmentar(imagem_processada, kernel, iteracoes)
    resultado = classificar_estado(mascaras_limpas)
    mostrar(
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


def processar_video(caminho, coordenadas, filtro, kernel=5, iteracoes=1):
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
        mascaras, mascaras_limpas = segmentar(
            imagem_processada, kernel, iteracoes
        )
        resultado = classificar_estado(mascaras_limpas)
        mostrar(
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
        default=5,
        help="Tamanho do kernel morfológico (padrão: 5).",
    )
    parser.add_argument(
        "--iteracoes",
        type=int,
        default=1,
        help="Iterações das operações morfológicas (padrão: 1).",
    )
    args = parser.parse_args()

    if not args.entrada.is_file():
        parser.error("arquivo inexistente ou inválido")

    try:
        if not processar_imagem(
            args.entrada, args.roi, args.filtro, args.kernel, args.iteracoes
        ):
            processar_video(
                args.entrada, args.roi, args.filtro, args.kernel, args.iteracoes
            )
    except ValueError as erro:
        parser.error(str(erro))
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
