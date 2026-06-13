"""Entrada, processamento e detecção do estado do semáforo."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from deteccao import classificar_estado
from processamento import (
    aplicar_regioes_esperadas,
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
    altura_processada = detectada.shape[0]
    for y in (altura_processada // 3, 2 * altura_processada // 3):
        cv2.line(detectada, (0, y), (detectada.shape[1], y), (180, 180, 180), 1)
    if resultado["regiao"]:
        rx, ry, rw, rh = resultado["regiao"]
        cv2.rectangle(detectada, (rx, ry), (rx + rw, ry + rh), cor, 2)

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


def segmentar(imagem, kernel, iteracoes):
    mascaras = criar_mascaras(converter_hsv(imagem))
    mascaras = aplicar_regioes_esperadas(mascaras)
    return mascaras, limpar_mascaras(mascaras, kernel, iteracoes)


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
