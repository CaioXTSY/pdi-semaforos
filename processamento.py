"""Entrada, pré-processamento e segmentação das imagens."""

import argparse
from pathlib import Path

import cv2
import numpy as np

LARGURA_MAXIMA = 960
AMPLIACAO_PADRAO = 4
KERNEL_PADRAO = 3

# O matiz (H) do OpenCV varia de 0 a 179.
LIMITES_HSV = {
    "vermelho": [
        ((0, 80, 60), (12, 255, 255)),
        ((168, 80, 60), (179, 255, 255)),
    ],
    "amarelo": [((15, 80, 70), (38, 255, 255))],
    "verde": [((38, 60, 50), (90, 255, 255))],
}

ZONAS_VERTICAIS = {
    "vermelho": (0.00, 0.42),
    "amarelo": (0.25, 0.72),
    "verde": (0.55, 1.00),
}


def ler_argumentos():
    parser = argparse.ArgumentParser(description="Detecta o estado de um semáforo.")
    parser.add_argument("entrada", type=Path)
    parser.add_argument("--roi", nargs=4, type=int, metavar=("X", "Y", "L", "A"))
    parser.add_argument(
        "--filtro", choices=("mediana", "gaussiano"), default="mediana"
    )
    parser.add_argument("--kernel", type=int, default=KERNEL_PADRAO)
    parser.add_argument("--iteracoes", type=int, default=1)
    parser.add_argument("--ampliacao", type=float, default=AMPLIACAO_PADRAO)
    parser.add_argument(
        "--saida", type=Path, default=Path("resultados/resultados.csv")
    )
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


def redimensionar(frame, largura=LARGURA_MAXIMA):
    """Limita a largura do quadro sem alterar sua proporção."""
    if frame.shape[1] <= largura:
        return frame
    escala = largura / frame.shape[1]
    return cv2.resize(frame, None, fx=escala, fy=escala)


def selecionar_roi(frame, coordenadas=None):
    """Recorta a ROI informada ou selecionada com o mouse."""
    if coordenadas is None:
        coordenadas = cv2.selectROI("Selecione o semaforo", frame, False)
        cv2.destroyWindow("Selecione o semaforo")

    x, y, largura, altura = map(int, coordenadas)
    limite_x, limite_y = x + largura, y + altura
    if x < 0 or y < 0 or largura <= 0 or altura <= 0:
        raise ValueError("Região de interesse inválida.")
    if limite_x > frame.shape[1] or limite_y > frame.shape[0]:
        raise ValueError("Região de interesse fora da imagem.")
    return frame[y:limite_y, x:limite_x], (x, y, largura, altura)


def preprocessar(imagem, filtro="mediana", ampliacao=AMPLIACAO_PADRAO):
    """Amplia a ROI, melhora seu contraste e reduz ruídos."""
    if ampliacao <= 0:
        raise ValueError("A ampliação deve ser maior que zero.")

    imagem = cv2.resize(
        imagem, None, fx=ampliacao, fy=ampliacao, interpolation=cv2.INTER_CUBIC
    )
    hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = cv2.createCLAHE(2.0, (8, 8)).apply(hsv[:, :, 2])
    imagem = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    filtros = {
        "gaussiano": lambda img: cv2.GaussianBlur(img, (3, 3), 0),
        "mediana": lambda img: cv2.medianBlur(img, 3),
    }
    if filtro not in filtros:
        raise ValueError("Filtro deve ser 'gaussiano' ou 'mediana'.")
    return filtros[filtro](imagem)


def criar_mascaras(imagem):
    """Segmenta vermelho, amarelo e verde no espaço HSV."""
    hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)
    mascaras = {}

    for cor, faixas in LIMITES_HSV.items():
        mascara = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for minimo, maximo in faixas:
            faixa = cv2.inRange(hsv, np.array(minimo), np.array(maximo))
            mascara = cv2.bitwise_or(mascara, faixa)
        mascaras[cor] = mascara
    return mascaras


def limitar_zonas(mascaras):
    """Mantém cada cor na faixa vertical esperada do semáforo."""
    altura = next(iter(mascaras.values())).shape[0]
    resultado = {}

    for cor, mascara in mascaras.items():
        inicio, fim = ZONAS_VERTICAIS[cor]
        y1, y2 = int(altura * inicio), int(altura * fim)
        resultado[cor] = np.zeros_like(mascara)
        resultado[cor][y1:y2] = mascara[y1:y2]
    return resultado


def limpar_mascara(mascara, tamanho_kernel, iteracoes):
    """Remove pequenos ruídos com fechamento e abertura."""
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


def segmentar(imagem, tamanho_kernel=KERNEL_PADRAO, iteracoes=1):
    """Cria as máscaras originais e suas versões sem ruído."""
    mascaras = limitar_zonas(criar_mascaras(imagem))
    limpas = {
        cor: limpar_mascara(mascara, tamanho_kernel, iteracoes)
        for cor, mascara in mascaras.items()
    }
    return mascaras, limpas
